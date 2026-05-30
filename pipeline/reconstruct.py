import cv2
import numpy as np
import open3d as o3d
import logging
from algorithm.match import FeatureMatcher
from algorithm.errors import *
from management.worldmap import Map

logger = logging.getLogger(__name__)

class Reconstructor:
    """Legacy temporal reconstruction prototype.

    The current unordered pipeline is implemented in incremental_unordered.py.
    This module is kept as historical reference and is not the main entry point.
    """

    def __init__(self, worldmap:Map, matcher:FeatureMatcher, frame_dir):

        self.map = worldmap
        self.matcher = matcher
        self.frame_dir = frame_dir

        # Reserved for legacy configuration.
        self.parameter = None

    def run(self):
        """Run the legacy temporal reconstruction loop."""
        logger.info("启动 legacy temporal 重建流水线")
        logger.info(f"图像路径: {self.frame_dir}")

        try:
            self.initialize_worldmap(self.frame_dir)
            logger.info(f"初始化成功。已注册帧: {list(self.map._registered_ids)}")
            logger.info(f"当前地图点数: {len(self.map._points)}")
        except Exception as e:
            logger.exception(f"初始化失败: {e}")
            return

        while len(self.map.unregistered_frames) > 0:
            
            next_frame_id = self.map.unregistered_frames[0]
            
            logger.info(f"正在尝试注册帧 ID: {next_frame_id}")
            
            success = self.add_next_frame()

            if success:
                logger.info(f"帧 {next_frame_id} 注册完成。当前总地图点数: {len(self.map._points)}")
            else:
                logger.warning(f"帧 {next_frame_id} 注册失败，尝试跳过。")
                
                # Legacy fallback: mark the frame as processed to avoid a loop.
                self.map._registered_ids.add(next_frame_id) 

        logger.info(f"重建结束。共注册 {len(self.map._registered_ids)} 帧, 生成 {len(self.map._points)} 个地图点。")

    def initialize_worldmap(self, frame_dir):

        self.map.load_frame_dir(frame_dir)

        # Legacy temporal mode always starts from the first two frames.
        frame1_idx = self.map.unregistered_frames[0]
        frame2_idx = self.map.unregistered_frames[1]

        self._extract_single_frame(frame1_idx)
        self._extract_single_frame(frame2_idx)
        logger.info("初始帧特征提取完成。")

        _, inlier_matches,_ ,model_type, _, _ = self._match_2d_pair(frame1_idx, frame2_idx)
        if model_type != "F":
            raise ValueError(f"[Reconstructor]: model type is not general 3D")
        logger.info(f"初始帧匹配完成，GRIC 选择模型: {model_type}")

        D_inlier_matches = self._register_initial_frames(frame1_idx, frame2_idx, inlier_matches)
        logger.info(f"初始位姿恢复得到 {len(D_inlier_matches)} 个深度有效匹配。")

        candidates = self._triangulate_between_frames(frame1_idx, frame2_idx, D_inlier_matches)
        logger.info(f"生成 {len(candidates)} 个候选地图点。")

        new_add_count = self._register_tri_candidates(frame1_idx,frame2_idx, candidates)
        logger.info(f"注册 {new_add_count} 个新地图点。")

    def add_next_frame(self):

        unregistered = self.map.unregistered_frames
        if not unregistered:
            logger.info("没有更多待注册帧。")
            return False
            
        new_frame_idx = unregistered[0]
        # Legacy temporal mode matches only against the most recent registered frame.
        last_frame_idx = list(self.map._registered_ids)[-1]

        logger.info(f"处理帧 {new_frame_idx}，参考帧 {last_frame_idx}")

        self._extract_single_frame(new_frame_idx)

        _, inlier_matches,_ ,model_type, _, _ = self._match_2d_pair(last_frame_idx, new_frame_idx)
        if model_type != "F":
            raise ValueError(f"[Reconstructor]: model type is not general 3D")

        pts_3d, pts_2d, pts_3d_ids, pts_2d_ids = self._find_2d_3d_correspondences(
            last_frame_idx, new_frame_idx, inlier_matches
        )

        if len(pts_3d) < 7:
            logger.warning(f"2D-3D 对应过少 ({len(pts_3d)})。")
            return False

        try:
            self._register_frame_pnp(new_frame_idx, pts_3d, pts_2d, pts_3d_ids, pts_2d_ids)
        except ValueError as e:
            logger.warning(f"PnP 注册失败: {e}")
            return False

        candidates = self._triangulate_between_frames(last_frame_idx, new_frame_idx, inlier_matches)

        new_points_count = self._register_tri_candidates(last_frame_idx, new_frame_idx, candidates)
        logger.info(f"帧 {new_frame_idx} 注册完成，新增 {new_points_count} 个点。")
        return True 
    def init_worldmap_and_viewgraph(self, frame_dir):

        self.map.load_frame_dir(frame_dir)

        self._extract_all_frames()

        self._build_view_graph()

        self._get
    
    def init_pose(self):
        pass

    def add_next_frame(self):
        pass

    def _extract_all_frames(self):
        """Extract features for all frames."""
        frames = list(self.map.all_frames())
        if not frames:
            raise ValueError(f"[Reconstructor]: extract all frames failed, world map has no frame!")
        for frame in frames:
            self.matcher.extract(frame)

    def _extract_single_frame(self, frame_idx):
        """Extract features for one frame."""
        frame = self.map.get_frame(frame_idx)
        self.matcher.extract(frame)

    def _match_2d_pair(self, frame1_idx, frame2_idx):
        """Match two frames and return the selected two-view geometry."""

        frame1 = self.map.get_frame(frame1_idx)
        frame2 = self.map.get_frame(frame2_idx)
        matrix, inlier_matches, inlier_ratio, model_type, gric_f, gric_h = self.matcher.match_2d_pair(frame1, frame2)

        return matrix, inlier_matches, inlier_ratio, model_type, gric_f, gric_h
    
    def _build_view_graph(self):
        
        frames = list(self.map.all_frames())
        for i in range(len(frames)):
            for j in range(i+1,len(frames)):
                try:
                    _, inlier_matches, inlier_ratio, model_type, gric_f, gric_h = self.matcher.match_2d_pair(frames[i], frames[j])
                    self.map.add_view_graph_edge(frames[i].idx, frames[j].idx, 
                                                 inlier_matches, inlier_ratio, 
                                                 model_type, gric_f, gric_h)
                except InsufficientMatchesError:
                    continue

                
    def _register_initial_frames(self, frame1_idx:int, frame2_idx:int, F_inlier_matches:list[cv2.DMatch]):
        """Recover the initial relative pose and return cheirality inliers."""

        frame1 = self.map.get_frame(frame1_idx)
        frame2 = self.map.get_frame(frame2_idx)

        pts1 = np.float32([frame1.kps[m.queryIdx].pt for m in F_inlier_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([frame2.kps[m.trainIdx].pt for m in F_inlier_matches]).reshape(-1, 1, 2)

        E, mask_E = cv2.findEssentialMat(pts1, pts2, frame1.camera.K, method=cv2.RANSAC, threshold = 3, prob=0.999)
        retval, R, t, mask_Depth = cv2.recoverPose(E, pts1, pts2, cameraMatrix=frame1.camera.K, mask = mask_E)

        if not retval:
            raise ValueError(f"[Reconstruct]: compute essential matrix failed!")

        I_R = np.eye(3)
        I_t = np.zeros((3, 1))

        self.map.register_frame(frame1_idx,I_R,I_t)
        self.map.register_frame(frame2_idx,R,t)

        D_inlier_matches = []

        matches_mask = mask_Depth.ravel().tolist()
        for i, match in enumerate(F_inlier_matches):
            if matches_mask[i]==1:
                D_inlier_matches.append(match)

        return D_inlier_matches
    
    def _triangulate_between_frames(self, frame1_idx:int, frame2_idx:int, inlier_matches:list[cv2.DMatch]):
        """Triangulate unmatched feature pairs between two registered frames."""

        frame1 = self.map.get_frame(frame1_idx)
        frame2 = self.map.get_frame(frame2_idx)

        P1 = frame1.get_proj_matrix()
        P2 = frame2.get_proj_matrix()

        tri_matches = []

        # Existing point observations are extended; only new-new pairs are triangulated.
        for m in inlier_matches:

            id1, id2 = m.queryIdx, m.trainIdx 
            
            p1 = frame1.feature_2_point.get(id1)
            p2 = frame2.feature_2_point.get(id2)

            if p1 is not None and p2 is None:
                self.map.add_observation(p1, frame2_idx, id2)
            elif p1 is None and p2 is not None:
                self.map.add_observation(p2, frame1_idx, id1)
            elif p1 is not None and p2 is not None:
                continue
            else:
                tri_matches.append(m)
            
        pts1 = np.float32([frame1.kps[m.queryIdx].pt for m in tri_matches]).reshape(-1, 2).T
        pts2 = np.float32([frame2.kps[m.trainIdx].pt for m in tri_matches]).reshape(-1, 2).T

        points4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

        if points4d is None:
            raise TriangulateError(f"[reconstructor]: triangulation bewteen frame {frame1_idx} and {frame2_idx} failed")

        xyz = points4d[:3, :]
        w = points4d[3:, :]
        points_normalized = (xyz / w).T

        
        o1 = frame1.get_center().flatten()
        o2 = frame2.get_center().flatten()
        R1, t1 = frame1.R, frame1.t 
        R2, t2 = frame2.R, frame2.t
        MIN_PARALLAX_DEG = 1.0
        max_cos_threshold = np.cos(np.deg2rad(MIN_PARALLAX_DEG))

        candidates = []
        for i, x in enumerate(points_normalized):

            p_c1 = R1 @ x.reshape(3,1) + t1
            p_c2 = R2 @ x.reshape(3,1) + t2
            if p_c1[2] <= 0 or p_c2[2] <= 0:
                continue

            o1x = x - o1
            o2x = x - o2
            norm_o1x = np.linalg.norm(o1x)
            norm_o2x = np.linalg.norm(o2x)
            if norm_o1x < 1e-6 or norm_o2x < 1e-6:
                continue
            cos_theta = np.dot(o1x, o2x) / (norm_o1x * norm_o2x)
            if cos_theta > max_cos_threshold:
                continue

            match = tri_matches[i]
            f1_feat_idx = match.queryIdx
            f2_feat_idx = match.trainIdx

            u, v = map(int, frame1.kps[f1_feat_idx].pt) 
            bgr = frame1.get_color(u,v) 
            rgb = bgr[::-1] / 255.0

            new_point = self.map.create_point(x,color=rgb)
            candidates.append((new_point, f1_feat_idx, f2_feat_idx))    

        return candidates
    
    def _find_2d_3d_correspondences(self, last_frame_idx: int, new_frame_idx: int, F_inlier_matches: list[cv2.DMatch]):
        """Find 2D-3D correspondences for PnP registration."""
        frame_last = self.map.get_frame(last_frame_idx)
        frame_new = self.map.get_frame(new_frame_idx)

        pts_3d = [] 
        pts_2d = [] 
        pts_3d_ids = []
        pts_2d_ids = []

        for match in F_inlier_matches:
            
            last_feat_idx = match.queryIdx
            new_feat_idx = match.trainIdx
            point_idx = frame_last.get_observed_point(last_feat_idx)
            
            if point_idx is not None:
                point = self.map.get_point(point_idx)
                if point is not None:
                    pts_3d.append(point.position3d)
                    pts_2d.append(frame_new.kps[new_feat_idx].pt)
                    pts_3d_ids.append(point_idx)
                    pts_2d_ids.append(new_feat_idx)

        return np.array(pts_3d, dtype=np.float64), np.array(pts_2d, dtype=np.float64), pts_3d_ids, pts_2d_ids

    def _register_frame_pnp(self, new_frame_idx:int, pts_3d:np.ndarray, pts_2d:np.ndarray, pts_3d_ids:list[int], pts_2d_ids:list[int]):
        """Register a frame pose and attach PnP inlier observations."""

        frame_new = self.map.get_frame(new_frame_idx)
        retval, rvec, tvec, PnP_inlier_matches = cv2.solvePnPRansac(
                                        pts_3d, pts_2d, frame_new.camera.K, distCoeffs=None, 
                                        flags=cv2.SOLVEPNP_ITERATIVE, iterationsCount=100, 
                                        reprojectionError=8.0, confidence=0.99
                                        )
        
        if (not retval) or  (PnP_inlier_matches is None):
            raise ValueError(f"[Reconstruct] compute PnP_iterative failed!")
        

        R, _ = cv2.Rodrigues(rvec)
        self.map.register_frame(new_frame_idx, R, tvec)

        for i in range(len(PnP_inlier_matches)):

            inlier_idx = PnP_inlier_matches[i][0]
            point_idx = pts_3d_ids[inlier_idx]
            feature_idx = pts_2d_ids[inlier_idx]
            
            self.map.add_observation(point_idx, new_frame_idx, feature_idx)

        return  PnP_inlier_matches
    
    def _register_tri_candidates(self, last_frame_idx, new_frame_idx, candidates):

        new_points_count = 0
        for point, feat1_idx, feat2_idx in candidates:
            point_idx = self.map.register_point(point)
            
            self.map.add_observation(point_idx, last_frame_idx, feat1_idx)
            self.map.add_observation(point_idx, new_frame_idx, feat2_idx)
            new_points_count += 1
        
        return new_points_count
        


















    
        


