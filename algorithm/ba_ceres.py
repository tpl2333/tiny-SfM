import numpy as np
import collections
import cv2
import logging
logger = logging.getLogger(__name__)

from management.worldmap import Map
from management.trackmanager import TrackManager

from build.Release import ba_core  # 确保你编译好的 .pyd 或 .so 在路径下



class BundleAdjuster:
    def __init__(self, worldmap:Map, trackmanager:TrackManager):
        self.map = worldmap
        self.tm = trackmanager

    def run_global_ba(self, fixed_frame_idx, is_fixed_focal = False):
        """ 优化地图中所有已注册的帧和所有地图点 """
        frame_ids = list(self.map._registered_ids)
        point_ids = list(self.map._points.keys())
        self._optimize(frame_ids, point_ids, fixed_frame_idx, is_fixed_focal, mode = "Global")

    def run_local_ba(self, window_size = 5):
        reg_seq = self.map._registered_sequence
        if len(reg_seq) < 3:
            return

        # --- 1. 确定活跃相机 (Active Frames) ---
        # 依然取最近注册的 N 帧，因为它们最需要被“校准”
        active_frame_ids = reg_seq[-window_size:]
        active_set = set(active_frame_ids)

        # --- 2. 确定活跃点 (Active Points) 与 潜在锚点候选 ---
        active_point_ids = set()
        anchor_counts = collections.Counter()

        for f_idx in active_frame_ids:
            # 获取该帧所有带 3D 点的特征
            feat_indices, pt_indices = self.tm.get_2d_3d_pairs(f_idx)
            for p_idx in pt_indices:
                active_point_ids.add(p_idx)
                
                # 寻找这个点的所有观测者，作为锚点候选
                track = self.tm._tracks.get(self.map._point_to_track[p_idx])
                if track:
                    for obs_f_idx, _ in track.observations:
                        if obs_f_idx not in active_set and obs_f_idx in self.map._registered_ids:
                            anchor_counts[obs_f_idx] += 1

        if not active_point_ids:
            return

        # --- 3. 确定固定锚点 (Anchor Frames) ---
        # 选取共视点数最多的前 2 帧作为固定帧
        most_common_anchors = anchor_counts.most_common(2)
        fixed_frame_ids = [f_idx for f_idx, count in most_common_anchors]
        
        # 兜底：如果没找到共视锚点（比如刚开始），固定第一帧
        if not fixed_frame_ids:
            fixed_frame_ids = [reg_seq[0]]

        # --- 4. 执行优化 ---
        all_involved_frames = list(active_set | set(fixed_frame_ids))
        point_ids = list(active_point_ids)

        self._optimize(
            frame_ids=all_involved_frames,
            point_ids=point_ids,
            fixed_frame_ids=fixed_frame_ids,
            is_fixed_focal=True, # 局部 BA 严禁动焦距
            mode="Local"
        )


    def _optimize(self, frame_ids, point_ids, fixed_frame_ids, is_fixed_focal = True, mode="BA"):
        if not frame_ids or not point_ids:
            return

        # --- 1. 建立 ID 到连续索引的映射 (Ceres 需要) ---
        f_idx_to_ceres_idx = {f_idx: i for i, f_idx in enumerate(frame_ids)}
        fixed_frame_ceres_ids = [f_idx_to_ceres_idx[f_idx] for f_idx in fixed_frame_ids]

        # --- 2. 提取相机参数 [N, 6] ---
        # Snavely 格式: [r1, r2, r3, t1, t2, t3]
        cameras = []
        for f_idx in frame_ids:
            frame = self.map.get_frame(f_idx)
            rvec, _ = cv2.Rodrigues(frame.R)
            cam_param = np.hstack([rvec.flatten(), frame.t.flatten()])
            cameras.append(cam_param)
        cameras = np.array(cameras, dtype=np.float64)

        # --- 3. 提取地图点坐标 [M, 3] ---
        points = np.array([self.map.get_point(p_id).position3d for p_id in point_ids], dtype=np.float64)

        # --- 4. 提取观测数据 ---
        obs_data = []
        cam_ceres_indices = []
        pt_ceres_indices = []

        for p_ceres_idx, p_idx in enumerate(point_ids):
            track_idx = self.map.get_point(p_idx).track_idx
            track = self.tm._tracks[track_idx]
            for f_idx, feat_idx in track.observations:
                if f_idx in f_idx_to_ceres_idx:
                    # 记录观测到的像素坐标
                    obs_data.append(self.map.get_frame(f_idx).kps[feat_idx].pt)
                    cam_ceres_indices.append(f_idx_to_ceres_idx[f_idx])
                    pt_ceres_indices.append(p_ceres_idx)

        obs_data = np.array(obs_data, dtype=np.float64)
        cam_ceres_indices = np.array(cam_ceres_indices, dtype=np.int32)
        pt_ceres_indices = np.array(pt_ceres_indices, dtype=np.int32)

        fixed_frame_ceres_ids = np.array(fixed_frame_ceres_ids, dtype=np.int32)

        # 拿到主点 (cx, cy) 与 焦距 f
        K = self.map.get_intrisics()
        cx, cy = K[0, 2], K[1, 2]
        shared_focal = np.array([K[0,0]], dtype=np.float64)

        logger.info(f"[{mode} BA] 优化中: {len(frame_ids)}个相机, {len(point_ids)}个点, {len(obs_data)}次观测...")

        # --- 5. 调用 C++ 后端 ---
        # 直接修改 cameras 和 points 数组的内容

        cameras = np.ascontiguousarray(cameras)
        points = np.ascontiguousarray(points)
        shared_focal = np.ascontiguousarray(shared_focal)
        obs_data = np.ascontiguousarray(obs_data)
        cam_ceres_indices = np.ascontiguousarray(cam_ceres_indices)
        pt_ceres_indices = np.ascontiguousarray(pt_ceres_indices)
        fixed_frame_ceres_ids = np.ascontiguousarray(fixed_frame_ceres_ids)

        ba_core.solve_ba_shared_focal(
            cameras, points, shared_focal,
            obs_data,
            cam_ceres_indices, pt_ceres_indices,
            fixed_frame_ceres_ids,
            is_fixed_focal,
            cx, cy
        )

        # --- 6. 将优化结果写回 WorldMap ---
        for i, f_idx in enumerate(frame_ids):
            opt_cam = cameras[i]
            R_new, _ = cv2.Rodrigues(opt_cam[:3])
            t_new = opt_cam[3:6].reshape(3, 1)
            # 更新位姿和焦距 (如果需要)
            self.map.get_frame(f_idx).set_pose(R_new, t_new)
        
        if not is_fixed_focal:
            self.map.set_focal(shared_focal[0])
        
        for i, p_id in enumerate(point_ids):      
            new_pos = points[i]
            if np.isnan(new_pos).any() or np.isinf(new_pos).any():
                continue 
            self.map.get_point(p_id).set_position3d(new_pos)

        logger.info(f"[{mode} BA] 优化完成！")