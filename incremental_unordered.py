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
    """Incremental unordered SfM pipeline.

    Pipeline:
    1. Load images and extract features.
    2. Exhaustively match frame pairs and build a view graph.
    3. Merge pairwise matches into global feature tracks.
    4. Initialize from a strong seed pair, then register frames with PnP.
    5. Triangulate new tracks, run BA, and prune unstable points.
    """

    def __init__(self, camera:Camera, img_dir="./data/frame"):

        self.worldmap = Map(camera)
        self.viewgraph = ViewGraph()
        self.trackmanager = TrackManager()

        self.matcher = FeatureMatcher()
        self.dataminer = DataMiner()
        self.mvgsolver = MvgSolver()
        self.ba = BundleAdjuster(self.worldmap, self.trackmanager)

        self.worldmap.load_frame_dir(img_dir)
        track_length_threshold = self.determine_track_threshold()

        frames = list(self.worldmap.all_frames())
        self.matcher.extract_all(frames)

        self.matcher.match_exhaustive(frames, self.viewgraph)

        threshold = self.determine_track_threshold()
        self.trackmanager.build_from_viewgraph(self.viewgraph, threshold)

        logger.info(f"初始化成功")

        self.canonical_f1_idx = None
        self.canonical_f2_idx = None

    def determine_track_threshold(self):
        """Require longer tracks when more images are available."""
        num_frames = len(self.worldmap._frames)
        if num_frames < 6:
            return 2
        elif num_frames < 15:
            return 3
        else:
            return 4
    
    def run(self):
        """Run initialization, incremental registration, BA, and cleanup."""
        seed = self.dataminer.find_best_seed(self.viewgraph, self.worldmap)
        frame1_idx, frame2_idx, _ = seed
        logger.info(f"选择 帧{frame1_idx} 与 帧{frame2_idx} 作为初始化帧")

        frame1 = self.worldmap.get_frame(frame1_idx)
        frame2 = self.worldmap.get_frame(frame2_idx)
        edge = self.viewgraph.get_edge(frame1_idx, frame2_idx)

        R, t, D_inlier_matches = self.mvgsolver.get_initial_pose(frame1, frame2, edge)
        logger.info(f"经过本质矩阵与深度检测，初始化帧对产生了 {len(D_inlier_matches)} 个三角化匹配点")

        self.canonical_f1_idx = frame1_idx
        self.canonical_f2_idx = frame2_idx
        self.worldmap.register_frame(frame1_idx)
        self.worldmap.register_frame(frame2_idx, R, t)

        K = self.worldmap.get_intrisics()
        obs_tracks, obs_matches, tri_tracks, tri_matches = self.trackmanager.classify_matches(frame1_idx, frame2_idx, D_inlier_matches)
        point_info = self.mvgsolver.triangulate(frame1, frame2, tri_tracks, tri_matches, K)

        point_indice = self.worldmap.create_points_from_info(point_info)
        self.trackmanager.update_track_state(point_info, point_indice)


        while True:

            next_frame_idx, count = self.dataminer.find_next_best_frame(self.worldmap, self.viewgraph, self.trackmanager)
            
            if next_frame_idx is None or count < 8:
                logger.info("没有合适的候选帧或所有帧已注册，重建结束。")
                break
                
            logger.info(f" 下一个目标: 帧{next_frame_idx} (拥有 {count} 个 2D-3D 对应)")

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

            all_neighbors = self.viewgraph.get_connected_frames(next_frame_idx)
            registered_neighbors = all_neighbors & self.worldmap.registered_frame_set

            for rb_idx in registered_neighbors:
                edge = self.viewgraph.get_edge(next_frame_idx, rb_idx)
                if edge is None: continue
               
                # Edge storage is ordered by frame id; align matches to (new, neighbor).
                matches_aligned = edge.matches[:, [1, 0]] if next_frame_idx > rb_idx else edge.matches
                
                _, _, tri_tracks_ids, tri_matches = self.trackmanager.classify_matches(next_frame_idx, rb_idx, matches_aligned)
                if len(tri_matches) == 0: continue
                
                f_new = self.worldmap.get_frame(next_frame_idx)
                f_nb = self.worldmap.get_frame(rb_idx)
                K = self.worldmap.get_intrisics()

                point_info_to_add = []

                for track_idx, m in zip(tri_tracks_ids, tri_matches):
                    track = self.trackmanager.get_track_from_idx(track_idx)
                    pt_new = f_new.kps[m[0]].pt
                    pt_nb = f_nb.kps[m[1]].pt

                    # Tentative point from the current frame pair.
                    pt3d_temp = self.mvgsolver.triangulate_simple(
                        f_new.R, f_new.t, pt_new,
                        f_nb.R, f_nb.t, pt_nb, K
                    )

                    # Validate the tentative point against all registered observations.
                    is_ok, max_parallax, best_pair = self.mvgsolver.verify_multi_view_consensus(
                        pt3d_temp, track, self.worldmap
                    )

                    if is_ok and max_parallax > 2.0:
                        
                        # Re-triangulate with the widest accepted baseline.
                        p1, p2 = best_pair
                        pt3d_final = self.mvgsolver.triangulate_simple(
                            p1['R'], p1['t'], p1['pt'],
                            p2['R'], p2['t'], p2['pt'], K
                        )

                        best_color = None
                        max_saturation = -1.0

                        for obs_f_idx, obs_feat_idx in track.observations:
                            if obs_f_idx in self.worldmap.registered_frame_set:
                                obs_frame = self.worldmap.get_frame(obs_f_idx)
                                kp_pt = obs_frame.kps[obs_feat_idx].pt
                                u, v = int(kp_pt[0]), int(kp_pt[1])
                                
                                c = obs_frame.get_color(u, v)
                                c = c[::-1]/255.0
                                
                                # Prefer the least gray observation for point color.
                                saturation = np.max(c) - np.min(c)
                                
                                if saturation > max_saturation:
                                    max_saturation = saturation
                                    best_color = c
                        
                        if best_color is None:
                            kp_new = f_new.kps[m[0]]
                            bgr = f_new.get_color(kp_new.pt[0], kp_new.pt[1])
                            best_color = bgr[::-1] / 255.0

                        point_info_to_add.append((track_idx, pt3d_final, best_color))

                if point_info_to_add:
                    p_indices = self.add_new_points_safely(point_info_to_add)
                    logger.info(f" 帧 {next_frame_idx} 与 {rb_idx}: 通过审计，新增了 {len(p_indices)} 个点")
            
            if len(self.worldmap.registered_frame_set) % 5 == 0:
                logger.info("执行局部 BA 并清理地图...")
                global_fixed_frame_ids = [self.canonical_f1_idx, self.canonical_f2_idx]
                self.ba.run_global_ba(global_fixed_frame_ids,True)
                self.cleanup_map_points()


        logger.info(f"开始全局 BA")
        global_fixed_frame_ids = [self.canonical_f1_idx, self.canonical_f2_idx]
        self.ba.run_global_ba(global_fixed_frame_ids, True)
        self.cleanup_map_points()

    def add_new_points_safely(self, point_info):
        """Create map points and synchronize their owning tracks."""
        if not point_info: 
            return []
            
        point_indices = self.worldmap.create_points_from_info(point_info)
        self.trackmanager.update_track_state(point_info, point_indices)
        
        return point_indices

    def remove_bad_points_safely(self, point_indices):
        """Remove map points and mark their tracks as untriangulated again."""
        for pid in point_indices:
            track_idx = self.worldmap.remove_point(pid)
            if track_idx is not None:
                self.trackmanager.reset_track_state(track_idx)
    
    def cleanup_map_points(self, error_threshold=4.0):
        """Prune points with invalid depth or high mean reprojection error."""
        K = self.worldmap.get_intrisics()
        bad_point_ids = []
        
        for pid, point in self.worldmap._points.items():
            track = self.trackmanager.get_track_from_idx(point.track_idx)
            errors = []
            
            for f_idx, feat_idx in track.observations:
                if f_idx not in self.worldmap.registered_frame_set:
                    continue
                
                frame = self.worldmap.get_frame(f_idx)
                pt2d = frame.kps[feat_idx].pt
                
                err, depth = self.mvgsolver.calculate_repro_error(
                    point.position3d, frame.R, frame.t, K, pt2d
                )
                
                if depth <= 0:
                    errors = [float('inf')]
                    break
                errors.append(err)
            
            if not errors or np.mean(errors) > error_threshold:
                bad_point_ids.append(pid)

        if bad_point_ids:
            self.remove_bad_points_safely(bad_point_ids)
            logger.info(f"BA 后清理完成：剔除了 {len(bad_point_ids)} 个不合格点。")
            
import numpy as np
import open3d as o3d
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def visualize_reconstruction(reconstructor):
    """Visualize current map points and registered camera poses with Open3D."""
    logger.info("正在准备可视化数据...")
    
    points_3d = []
    colors = []
    for pt in reconstructor.worldmap._points.values():
        points_3d.append(pt.position3d)
        colors.append(pt.color)
    
    if not points_3d:
        logger.warning("地图中没有点云数据！")
        return

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(points_3d))
    pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

    geometries = [pcd]
    for f_idx in reconstructor.worldmap._registered_ids:
        frame = reconstructor.worldmap.get_frame(f_idx)
        R, t = frame.R, frame.t
        
        camera_center = -R.T @ t
        
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        
        T = np.eye(4)
        T[:3, :3] = R.T
        T[:3, 3] = camera_center.flatten()
        axis.transform(T)
        
        geometries.append(axis)

    logger.info(f"正在显示地图: {len(points_3d)} 个点, {len(geometries)-1} 个注册相机")
    o3d.visualization.draw_geometries(geometries, 
                                    window_name="SfM 重建结果",
                                    width=1280, height=720,
                                    left=50, top=50,
                                    mesh_show_back_face=True)

if __name__ == "__main__":
    from model.camera import Camera
    from incremental_unordered import Reconstructor

    img_dir = "./data/synthetic/ship/test"
    
    cam = Camera(height=800,width=800)
    cam.setup_by_guess(fov_scale=1.5625)

    try:
        recon = Reconstructor(cam, img_dir)
        
        recon.run()

        output_colmap_dir = "./output/synthetic/ship"
        recon.worldmap.save_as_colmap(output_colmap_dir, recon.trackmanager)
        
        visualize_reconstruction(recon)
        
    except Exception as e:
        logger.exception(f"重构过程中发生错误: {e}")
