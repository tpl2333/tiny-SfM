import numpy as np
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

    def run_global_ba(self, canonical_frame_idx):
        """ 优化地图中所有已注册的帧和所有地图点 """
        frame_ids = list(self.map._registered_ids)
        point_ids = list(self.map._points.keys())
        self._optimize(frame_ids, point_ids, canonical_frame_idx, "Global")

    def run_local_ba(self, new_frame_idx, window_size=5):
        """ 
        局部 BA：只优化新注册的帧及其看到的点。
        为了保持稳定，我们还会关联到能看到这些点的其他邻居帧。
        """
        # 1. 确定我们要优化的点：新帧能看到的点
        point_ids = []
        frame_obj = self.map.get_frame(new_frame_idx)
        for feat_idx in range(len(frame_obj.kps)):
            track = self.tm.get_track_from_feat(new_frame_idx, feat_idx)
            if track and track.is_triangulated:
                point_ids.append(track.mappoint_id)
        
        point_ids = list(set(point_ids)) # 去重

        # 2. 确定我们要优化的相机：看到这些点的所有已注册相机
        relevant_frame_ids = set([new_frame_idx])
        for p_id in point_ids:
            track_idx = self.map.get_point(p_id).track_idx
            track = self.tm._tracks[track_idx]
            for f_id, _ in track.observations:
                if f_id in self.map._registered_ids:
                    relevant_frame_ids.add(f_id)
        
        # 如果相机太多，这里可以根据 window_size 裁剪，目前先全放进去
        self._optimize(list(relevant_frame_ids), point_ids, "Local")

    def _optimize(self, frame_ids, point_ids, canonical_frame_idx, mode="BA"):
        if not frame_ids or not point_ids:
            return

        # --- 1. 建立 ID 到连续索引的映射 (Ceres 需要) ---
        f_id_to_idx = {f_id: i for i, f_id in enumerate(frame_ids)}
        p_id_to_idx = {p_id: i for i, p_id in enumerate(point_ids)}

        # --- 2. 提取相机参数 [N, 7] ---
        # Snavely 格式: [r1, r2, r3, t1, t2, t3, f]
        cameras = []
        for f_id in frame_ids:
            frame = self.map.get_frame(f_id)
            rvec, _ = cv2.Rodrigues(frame.R)
            f = frame.camera.K[0, 0] # 假设 fx = fy
            cam_param = np.hstack([rvec.flatten(), frame.t.flatten(), f])
            cameras.append(cam_param)
        cameras = np.array(cameras, dtype=np.float64)

        # --- 3. 提取地图点坐标 [M, 3] ---
        points = np.array([self.map.get_point(p_id).position3d for p_id in point_ids], dtype=np.float64)

        # --- 4. 提取观测数据 ---
        obs_data = []
        cam_indices = []
        pt_indices = []

        for p_idx, p_id in enumerate(point_ids):
            track_idx = self.map.get_point(p_id).track_idx
            track = self.tm._tracks[track_idx]
            for f_id, feat_idx in track.observations:
                if f_id in f_id_to_idx:
                    # 记录观测到的像素坐标
                    obs_data.append(self.map.get_frame(f_id).kps[feat_idx].pt)
                    cam_indices.append(f_id_to_idx[f_id])
                    pt_indices.append(p_idx)

        obs_data = np.array(obs_data, dtype=np.float64)
        cam_indices = np.array(cam_indices, dtype=np.int32)
        pt_indices = np.array(pt_indices, dtype=np.int32)

        # 拿到主点 (cx, cy)
        K = self.map.get_intrisics()
        cx, cy = K[0, 2], K[1, 2]

        # 固定参考帧
        fixed_idx = f_id_to_idx[canonical_frame_idx]

        logger.info(f"[{mode} BA] 优化中: {len(frame_ids)}个相机, {len(point_ids)}个点, {len(obs_data)}次观测...")

        # --- 5. 调用 C++ 后端 ---
        # 注意：这里会直接修改 cameras 和 points 数组的内容
        ba_core.solve_ba(
            cameras, points, obs_data,
            cam_indices, pt_indices,
            cx, cy, fixed_idx
        )

        # --- 6. 将优化结果写回 WorldMap ---
        for i, f_id in enumerate(frame_ids):
            opt_cam = cameras[i]
            R_new, _ = cv2.Rodrigues(opt_cam[:3])
            t_new = opt_cam[3:6].reshape(3, 1)
            # 更新位姿和焦距 (如果需要)
            self.map.get_frame(f_id).set_pose(R_new, t_new)
            # 如果想更新 K，也可以在这里修改


        for i, p_id in enumerate(point_ids):
            self.map.get_point(p_id).set_position3d(points[i])

        logger.info(f"[{mode} BA] 优化完成！")