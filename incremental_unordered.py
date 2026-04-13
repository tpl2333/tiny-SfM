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

        self.canonical_f_idx = None

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
        seed = self.dataminer.find_best_seed(self.viewgraph)
        frame1_idx, frame2_idx, _ = seed
        logger.info(f"选择 帧{frame1_idx} 与 帧{frame2_idx} 作为初始化帧")

        # 1.2 计算本质矩阵，分解获取初始位姿
        frame1 = self.worldmap.get_frame(frame1_idx)
        frame2 = self.worldmap.get_frame(frame2_idx)
        edge = self.viewgraph.get_edge(frame1_idx, frame2_idx)

        R, t, D_inlier_matches = self.mvgsolver.get_initial_pose(frame1, frame2, edge)
        logger.info(f"经过本质矩阵与深度检测，初始化帧对产生了 {len(D_inlier_matches)} 个三角化匹配点")

        # 1.3 注册初始帧
        self.canonical_f_idx = frame1_idx
        self.worldmap.register_frame(frame1_idx)
        self.worldmap.register_frame(frame2_idx, R, t)

        # 1.4 三角化对应点
        obs_tracks, obs_matches, tri_tracks, tri_matches = self.trackmanager.classify_matches(frame1_idx, frame2_idx, D_inlier_matches)
        point_info = self.mvgsolver.triangulate(frame1, frame2, tri_tracks, tri_matches)

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

            # 2.3 三角化 
            all_neighbors = self.viewgraph.get_connected_frames(next_frame_idx)
            registered_neighbors = all_neighbors & self.worldmap.registered_frame_set

            for rb_idx in registered_neighbors:

                # 1. 直接从 ViewGraph 拿现成的匹配关系
                edge = self.viewgraph.get_edge(next_frame_idx, rb_idx)
                if edge is None: continue

                if next_frame_idx > rb_idx:
                    matches_aligned = edge.matches[:, [1, 0]] 
                else:
                    matches_aligned = edge.matches
                
                # 2. 过滤所有匹配中还未三角化的点
                _, _, tri_tracks, tri_matches = self.trackmanager.classify_matches(next_frame_idx, rb_idx, matches_aligned)
                
                if len(tri_matches) == 0: continue
                
                # 3. 使用 MvgSolver 进行三角化 
                # 传入新帧、邻居帧以及待三角化的匹配对
                f_new = self.worldmap.get_frame(next_frame_idx)
                f_nb = self.worldmap.get_frame(rb_idx)
                
                # point_info: [(track_idx, xyz, color), ...]
                point_info = self.mvgsolver.triangulate(f_new, f_nb, tri_tracks, tri_matches)
                
                # 4. 注册新点并更新轨迹状态e
                point_indices = self.worldmap.create_points_from_info(point_info)
                self.trackmanager.update_track_state(point_info, point_indices)
                
                logger.info(f"通过帧 {next_frame_idx} 与帧 {rb_idx} 的三角化，新增了 {len(point_indices)} 个地图点")
            
            # 2.4 可能的局部ba


        # 3 最后的全局ba
        logger.info(f"开始全局光束法平差")
        self.ba.run_global_ba(self.canonical_f_idx)

            
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