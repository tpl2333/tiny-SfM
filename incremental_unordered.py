import logging

from model.camera import Camera
from model.edge import EdgeData

from management.viewgraph import ViewGraph
from management.trackmanager import TrackManager
from management.worldmap import Map

from algorithm.match import FeatureMatcher
from algorithm.datamine import DataMiner
from algorithm.mvgsolver import MvgSolver
from algorithm.ba_ceres import BundleAdjuster
from algorithm.errors import *

class Reconstructor:
    def __init__(self, camera:Camera, img_dir="./data/frame"):

        # 数据
        self.worldmap = Map(camera)
        self.viewgraph = ViewGraph()
        self.trackmanager = TrackManager()
        # 算法
        self.matcher = FeatureMatcher()
        self.dataminer = DataMiner()
        self.mvgsolver = MvgSolver()
        self.ba = BundleAdjuster(self.worldmap, self.trackmanager)

        # 1. 初始化 Worldmap，加载图像
        self.worldmap.load_frame_dir(img_dir)
        track_length_threshold = self.determine_track_threshold()

        # 2. 对所有图像进行特征点提取
        frames = list(self.worldmap.all_frames())
        self.matcher.extract_all(frames)

        # 3. 暴力匹配并初始化viewgraph
        self.matcher.match_exhaustive(frames, self.viewgraph)

        # 4. 通过viewgraph初始化特征轨迹
        threshold = self.determine_track_threshold()
        self.trackmanager.build_from_viewgraph(self.viewgraph, threshold)

        logger.info(f"初始化成功")

        self.canonical_f1_idx = None
        self.canonical_f2_idx = None

    def determine_track_threshold(self):
        num_frames = len(self.worldmap._frames)
        if num_frames < 6:
            return 2
        elif num_frames < 15:
            return 3
        else:
            return 4
    
    def run(self):
        # 1. 初始化
        # 1.1 通过 dataminer 在 viewgraph 里找到最适合进行初始化的两帧
        seed = self.dataminer.find_best_seed(self.viewgraph, self.worldmap)
        frame1_idx, frame2_idx, _ = seed
        logger.info(f"选择 帧{frame1_idx} 与 帧{frame2_idx} 作为初始化帧")

        # 1.2 计算本质矩阵，分解获取初始位姿
        frame1 = self.worldmap.get_frame(frame1_idx)
        frame2 = self.worldmap.get_frame(frame2_idx)
        edge = self.viewgraph.get_edge(frame1_idx, frame2_idx)

        R, t, D_inlier_matches = self.mvgsolver.get_initial_pose(frame1, frame2, edge)
        logger.info(f"经过本质矩阵与深度检测，初始化帧对产生了 {len(D_inlier_matches)} 个三角化匹配点")

        # 1.3 注册初始帧
        self.canonical_f1_idx = frame1_idx
        self.canonical_f2_idx = frame2_idx
        self.worldmap.register_frame(frame1_idx)
        self.worldmap.register_frame(frame2_idx, R, t)

        # 1.4 三角化对应点
        K = self.worldmap.get_intrisics()
        obs_tracks, obs_matches, tri_tracks, tri_matches = self.trackmanager.classify_matches(frame1_idx, frame2_idx, D_inlier_matches)
        point_info = self.mvgsolver.triangulate(frame1, frame2, tri_tracks, tri_matches, K)

        point_indice = self.worldmap.create_points_from_info(point_info)
        self.trackmanager.update_track_state(point_info, point_indice)
        # 1.5 可能的初始化ba


        # 2. 增量式重建
        while True:

            # 2.1 挑选下一帧
            next_frame_idx, count = self.dataminer.find_next_best_frame(self.worldmap, self.viewgraph, self.trackmanager)
            
            if next_frame_idx is None or count < 30:
                logger.info("没有合适的候选帧或所有帧已注册，重建结束。")
                break
                
            logger.info(f" 下一个目标: 帧{next_frame_idx} (拥有 {count} 个 2D-3D 对应)")

            # 2.2 PnP 解算该帧的位姿并注册
            next_frame = self.worldmap.get_frame(next_frame_idx)
            feat_ids, pt_ids = self.trackmanager.get_2d_3d_pairs(next_frame_idx)

            pts_2d = np.float32([next_frame.kps[i].pt for i in feat_ids])
            pts_3d = np.float32([self.worldmap.get_point(i).position3d for i in pt_ids])
            K = self.worldmap.get_intrisics()

            R, t, _ = self.mvgsolver.get_pose_from_pnp_iter(pts_2d, pts_3d, K)

            if R is None:
                self.worldmap.add_failed_frame(next_frame_idx)
                continue

            self.worldmap.register_frame(next_frame_idx, R, t)

            logger.info(f" 帧{next_frame_idx} 通过 PnP 解算，获得位姿 {R} 和 {t}")

            # 2.3 增量三角化 
            all_neighbors = self.viewgraph.get_connected_frames(next_frame_idx)
            registered_neighbors = all_neighbors & self.worldmap.registered_frame_set

            for rb_idx in registered_neighbors:
                edge = self.viewgraph.get_edge(next_frame_idx, rb_idx)
                if edge is None: continue
               
                # 统一匹配方向
                matches_aligned = edge.matches[:, [1, 0]] if next_frame_idx > rb_idx else edge.matches
                
                # 过滤未三角化的点
                _, _, tri_tracks_ids, tri_matches = self.trackmanager.classify_matches(next_frame_idx, rb_idx, matches_aligned)
                if len(tri_matches) == 0: continue
                
                f_new = self.worldmap.get_frame(next_frame_idx)
                f_nb = self.worldmap.get_frame(rb_idx)
                K = self.worldmap.get_intrisics()

                point_info_to_add = [] # 存储通过严选的点

                # --- 核心变更：逐轨迹进行精英审计 ---
                for track_idx, m in zip(tri_tracks_ids, tri_matches):
                    track = self.trackmanager.get_track_from_idx(track_idx)
                    pt_new = f_new.kps[m[0]].pt
                    pt_nb = f_nb.kps[m[1]].pt

                    # A. 初步尝试：用当前这对“发现者”算一个临时坐标
                    pt3d_temp = self.mvgsolver.triangulate_simple(
                        f_new.R, f_new.t, pt_new,
                        f_nb.R, f_nb.t, pt_nb, K
                    )

                    # B. 全局审计（公投）：拿这个点问该轨迹里所有的老相机
                    is_ok, max_parallax, best_pair = self.mvgsolver.verify_multi_view_consensus(
                        pt3d_temp, track, self.worldmap
                    )

                    # C. 准入审查：不仅要大家都不反对(is_ok)，地基还得稳(视差角 > 2.0)
                    if is_ok and max_parallax > 2.0:
                        
                        # D. 择优重算：既然审计通过，用基线最好的那对相机重新解算最高精度坐标
                        p1, p2 = best_pair
                        pt3d_final = self.mvgsolver.triangulate_simple(
                            p1['R'], p1['t'], p1['pt'],
                            p2['R'], p2['t'], p2['pt'], K
                        )

                         #  E.颜色融合 
                        best_color = None
                        max_saturation = -1.0

                        for obs_f_idx, obs_feat_idx in track.observations:
                            if obs_f_idx in self.worldmap.registered_frame_set:
                                obs_frame = self.worldmap.get_frame(obs_f_idx)
                                kp_pt = obs_frame.kps[obs_feat_idx].pt
                                u, v = int(kp_pt[0]), int(kp_pt[1])
                                
                                # 获取原始颜色 [R, G, B] (确保是 0.0-1.0)
                                c = obs_frame.get_color(u, v)
                                c = c[::-1]/255.0
                                
                                # 计算饱和度近似值：max(R,G,B) - min(R,G,B)
                                # 这个值越大，说明颜色越鲜艳，越不是灰/白/黑
                                saturation = np.max(c) - np.min(c)
                                
                                # 如果这帧的颜色更鲜艳，就选它
                                if saturation > max_saturation:
                                    max_saturation = saturation
                                    best_color = c
                        
                        # 如果没选到（理论上不会），就用当前帧保底
                        if best_color is None:
                            kp_new = f_new.kps[m[0]]
                            best_color = f_new.get_color(int(kp_new.pt[0]), int(kp_new.pt[1]))

                        point_info_to_add.append((track_idx, pt3d_final, best_color))

                # --- 执行安全入库 ---
                if point_info_to_add:
                    p_indices = self.add_new_points_safely(point_info_to_add)
                    logger.info(f" 帧 {next_frame_idx} 与 {rb_idx}: 通过审计，新增了 {len(p_indices)} 个点")
            
            # # 2.4 可能的局部ba
            if len(self.worldmap.registered_frame_set) % 5 == 0:
                logger.info("执行局部 BA 并清理地图...")
                global_fixed_frame_ids = [self.canonical_f1_idx, self.canonical_f2_idx]
                self.ba.run_global_ba(global_fixed_frame_ids,True)        # 执行优化
                self.cleanup_map_points() # 执行大扫除


        # 3 最后的全局ba
        logger.info(f"开始全局 BA")
        global_fixed_frame_ids = [self.canonical_f1_idx, self.canonical_f2_idx]
        self.ba.run_global_ba(global_fixed_frame_ids, True)
        self.cleanup_map_points()

    def add_new_points_safely(self, point_info):
        """一站式：创建 3D 点并绑定特征轨迹"""
        if not point_info: 
            return [] # 确保即使没有点也返回空列表，而不是 None
            
        # 1. 在 WorldMap 中生成 3D 点索引
        point_indices = self.worldmap.create_points_from_info(point_info)
        
        # 2. 同步更新 TrackManager 中的 mappoint_idx 状态
        self.trackmanager.update_track_state(point_info, point_indices)
        
        return point_indices

    def remove_bad_points_safely(self, point_indices):
        """ 一站式：删除 3D 点并解锁特征轨迹（用于后续 BA 后的清理） """
        for pid in point_indices:
            track_idx = self.worldmap.remove_point(pid) # 需要你在 worldmap 实现这个
            if track_idx is not None:
                self.trackmanager.reset_track_state(track_idx)
    
    def cleanup_map_points(self, error_threshold=4.0):
        """
        BA 后的地图净化：
        1. 重新计算所有点的平均重投影误差
        2. 识别并删除误差过大的离群点
        3. 重置对应轨迹状态，允许其后续‘重新投胎’
        """
        K = self.worldmap.get_intrisics()
        bad_point_ids = []
        
        # 遍历地图中当前所有的 3D 点
        for pid, point in self.worldmap._points.items():
            track = self.trackmanager.get_track_from_idx(point.track_idx)
            errors = []
            
            # 计算该点在所有观测帧中的重投影误差
            for f_idx, feat_idx in track.observations:
                if f_idx not in self.worldmap.registered_frame_set:
                    continue
                
                frame = self.worldmap.get_frame(f_idx)
                pt2d = frame.kps[feat_idx].pt
                
                # 调用 mvgsolver 中我们重构好的原子校验工具
                err, depth = self.mvgsolver.calculate_repro_error(
                    point.position3d, frame.R, frame.t, K, pt2d
                )
                
                if depth <= 0: # 如果点优化到了相机背面，必删
                    errors = [float('inf')]
                    break
                errors.append(err)
            
            # 如果平均误差超过阈值，标记为待删除
            if not errors or np.mean(errors) > error_threshold:
                bad_point_ids.append(pid)

        # 执行对称删除
        if bad_point_ids:
            self.remove_bad_points_safely(bad_point_ids)
            logger.info(f"BA 后清理完成：剔除了 {len(bad_point_ids)} 个不合格点。")
            
import numpy as np
import open3d as o3d
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_reconstruction(reconstructor):
    """
    使用 Open3D 可视化当前地图中的点云和相机位姿
    """
    logger.info("正在准备可视化数据...")
    
    # 1. 提取点云数据
    points_3d = []
    colors = []
    for pt in reconstructor.worldmap._points.values():
        points_3d.append(pt.position3d)
        colors.append(pt.color)
    
    if not points_3d:
        logger.warning("地图中没有点云数据！")
        return

    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    # 2. 提取相机位姿并创建坐标轴
    geometries = [pcd]
    for f_idx in reconstructor.worldmap._registered_ids:
        frame = reconstructor.worldmap.get_frame(f_idx)
        R, t = frame.R, frame.t
        
        # 计算相机在世界坐标系下的中心：C = -R^T * t
        camera_center = -R.T @ t
        
        # 创建一个坐标轴代表相机位姿
        # size 参数控制坐标轴大小，可以根据场景尺度调整
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        # 将轴变换到相机在世界系下的位置和朝向
        # 注意：Open3D 的变换矩阵是 4x4
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = camera_center.flatten()
        axis.transform(T)
        
        geometries.append(axis)

    # 3. 启动可视化窗口
    logger.info(f"正在显示地图: {len(points_3d)} 个点, {len(geometries)-1} 个注册相机")
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="SfM 重建结果",
                                    width=1280, height=720,
                                    left=50, top=50,
                                    mesh_show_back_face=True)

if __name__ == "__main__":
    from model.camera import Camera
    from incremental_unordered import Reconstructor

    # 1. 设定参数（请根据你的实际路径修改）
    img_dir = "./data/frame" # 图像文件夹路径
    
    # 2. 定义相机内参 
    cam = Camera(height=800,width=800)
    cam.setup_by_guess()

    # 3. 启动重构流水线
    try:
        # 初始化 Reconstructor 会自动执行加载、提取、匹配和构建 Track
        recon = Reconstructor(cam, img_dir)
        
        # 运行初始化流程 (寻找种子、位姿恢复、三角化)
        recon.run()
        
        # 4. 调用可视化
        visualize_reconstruction(recon)
        
    except Exception as e:
        logger.exception(f"重构过程中发生错误: {e}")