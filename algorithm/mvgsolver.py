import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)

from model.frame import Frame
from model.edge import EdgeData

from management.viewgraph import ViewGraph
from management.trackmanager import TrackManager
from management.worldmap import Map

from algorithm.errors import *

class MvgSolver:
    """Multi-view geometry helpers used by the incremental pipeline."""
    
    def get_initial_pose(self, frame1:Frame, frame2:Frame, edge:EdgeData):
        """Recover the relative pose of the second seed frame.

        The returned pose follows the project convention X_cam = R * X_world + t.
        The first seed frame is assumed to be identity by the caller.
        """
        pts1 = np.float32([frame1.kps[m].pt for m in edge.query_indices]).reshape(-1, 1, 2)
        pts2 = np.float32([frame2.kps[m].pt for m in edge.train_indices]).reshape(-1, 1, 2)

        E, mask_E = cv2.findEssentialMat(pts1, pts2, frame1.camera.K, method=cv2.RANSAC, threshold = 3, prob=0.999)
        retval, R, t, mask_Depth = cv2.recoverPose(E, pts1, pts2, cameraMatrix=frame1.camera.K, mask = mask_E)

        if not retval:
            raise ValueError(f"compute essential matrix failed!")

        inlier_mask = (mask_Depth.ravel() == 1)
        D_inlier_matches = edge.matches[inlier_mask]

        return R, t, D_inlier_matches

    def triangulate(self, frame1:Frame, frame2:Frame, tri_tracks:list[int], tri_matches:np.ndarray, K):
        """Triangulate candidate tracks observed by two registered frames."""
        tri_matches = np.atleast_2d(tri_matches)
        if frame1.idx == 21:
            logger.debug(f"{frame1.idx}-{frame2.idx} 匹配数: {tri_matches.shape}")

        if len(tri_matches) == 0:
            return []

        P1 = frame1.get_proj_matrix()
        P2 = frame2.get_proj_matrix()
       
        pts1 = np.float32([frame1.kps[m].pt for m in tri_matches[:,0]]).reshape(-1, 2).T
        pts2 = np.float32([frame2.kps[m].pt for m in tri_matches[:,1]]).reshape(-1, 2).T

        points4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

        if points4d is None:
            raise TriangulateError(f"[reconstructor]: triangulation bewteen frame {frame1.idx} and {frame2.idx} failed")

        xyz = points4d[:3, :]
        w = points4d[3:, :]
        points_normalized = (xyz / w).T

        o1 = frame1.get_center().flatten()
        o2 = frame2.get_center().flatten()
        R1, t1 = frame1.R, frame1.t 
        R2, t2 = frame2.R, frame2.t
        MIN_PARALLAX_DEG = 1.0
        max_cos_threshold = np.cos(np.deg2rad(MIN_PARALLAX_DEG))
        repro_error_threshold = 2.0

        point_info = []
        for i, x in enumerate(points_normalized):

            # Keep points in front of both cameras.
            p_c1 = R1 @ x.reshape(3,1) + t1
            p_c2 = R2 @ x.reshape(3,1) + t2
            if p_c1[2] <= 0 or p_c2[2] <= 0:
                continue

            # Small parallax makes depth unstable.
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
            track_idx = tri_tracks[i]
            f1_feat_idx = match[0]
            f2_feat_idx = match[1]
            
            proj1 = K @ p_c1
            u1_proj, v1_proj = proj1[0, 0] / proj1[2, 0], proj1[1, 0] / proj1[2, 0]
            u1_obs, v1_obs = frame1.kps[f1_feat_idx].pt
            err1 = np.sqrt((u1_proj - u1_obs)**2 + (v1_proj - v1_obs)**2)

            proj2 = K @ p_c2
            u2_proj, v2_proj = proj2[0, 0] / proj2[2, 0], proj2[1, 0] / proj2[2, 0]
            u2_obs, v2_obs = frame2.kps[f2_feat_idx].pt
            err2 = np.sqrt((u2_proj - u2_obs)**2 + (v2_proj - v2_obs)**2)

            if err1 > repro_error_threshold or err2 > repro_error_threshold:
                continue

            u, v = map(int, frame1.kps[f1_feat_idx].pt) 
            bgr = frame1.get_color(u,v) 
            rgb = bgr[::-1] / 255.0

            point_info.append((track_idx, x, rgb ))    

        return point_info
    

    def calculate_repro_error(self, pt3d, R, t, K, pt2d):
        """Return pixel reprojection error and camera-space depth."""
        pc = R @ pt3d.reshape(3, 1) + t.reshape(3, 1)
        depth = pc[2, 0]
        if depth <= 1e-6:
            return float('inf'), depth
        
        p_img = K @ pc
        u, v = p_img[0, 0] / p_img[2, 0], p_img[1, 0] / p_img[2, 0]
        error = np.linalg.norm(np.array([u, v]) - pt2d)
        return error, depth

    def calculate_parallax(self, pt3d, R1, t1, R2, t2):
        """Return the viewing-angle parallax of one point in degrees."""
        c1 = -R1.T @ t1.reshape(3, 1)
        c2 = -R2.T @ t2.reshape(3, 1)
        v1, v2 = pt3d.reshape(3, 1) - c1, pt3d.reshape(3, 1) - c2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6: return 0.0
        cos_theta = np.dot(v1.T, v2) / (n1 * n2)
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def triangulate_simple(self, R1, t1, pt1, R2, t2, pt2, K):
        """Triangulate one matched pixel pair with two known poses."""
        P1 = K @ np.hstack((R1, t1.reshape(3, 1)))
        P2 = K @ np.hstack((R2, t2.reshape(3, 1)))
        
        pts1_raw = np.array([pt1[0], pt1[1]], dtype=np.float32).reshape(2, 1)
        pts2_raw = np.array([pt2[0], pt2[1]], dtype=np.float32).reshape(2, 1)
        
        pt4d = cv2.triangulatePoints(P1, P2, pts1_raw, pts2_raw)
        return pt4d[:3, 0] / pt4d[3, 0]

    def verify_multi_view_consensus(self, pt3d, track, worldmap, err_thresh=4.0):
        """Validate a tentative 3D point against all registered observations."""
        K = worldmap.get_intrisics()
        valid_poses = []
        
        # Reject if any registered observation disagrees with this point.
        for f_idx, feat_idx in track.observations:
            if f_idx not in worldmap._registered_ids:
                continue
            
            frame = worldmap.get_frame(f_idx)
            pt2d = frame.kps[feat_idx].pt
            
            err, depth = self.calculate_repro_error(pt3d, frame.R, frame.t, K, pt2d)
            if err > err_thresh or depth <= 0:
                return False, 0.0, None
            
            valid_poses.append({'R': frame.R, 't': frame.t, 'pt': pt2d, 'id': f_idx})

        if len(valid_poses) < 2:
            return False, 0.0, None

        # The best baseline is reused to recompute a more stable point.
        max_parallax = 0.0
        best_pair = None
        for i in range(len(valid_poses)):
            for j in range(i + 1, len(valid_poses)):
                angle = self.calculate_parallax(pt3d, valid_poses[i]['R'], valid_poses[i]['t'],
                                               valid_poses[j]['R'], valid_poses[j]['t'])
                if angle > max_parallax:
                    max_parallax = angle
                    best_pair = (valid_poses[i], valid_poses[j])
                    
        return True, max_parallax, best_pair
    def get_pose_from_pnp_iter(self, pts_2d: np.ndarray, pts_3d: np.ndarray, K):
        """Estimate a frame pose from 2D-3D correspondences using PnP RANSAC."""
        if pts_3d.shape[0] < 6:
            logger.error(f"PnP 输入点数不足 6 个 ({pts_3d.shape[0]})，无法进行有限对极解算")
            return None, None, None

        # OpenCV PnP is sensitive to dtype and memory layout.
        pts_3d = np.ascontiguousarray(pts_3d, dtype=np.float32).reshape(-1, 1, 3)
        pts_2d = np.ascontiguousarray(pts_2d, dtype=np.float32).reshape(-1, 1, 2)
        K = np.ascontiguousarray(K, dtype=np.float32)

        # First pass: robust pose hypothesis and outlier rejection.
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, distCoeffs=None, 
            flags=cv2.SOLVEPNP_SQPNP,       
            iterationsCount=500,
            reprojectionError=8.0,
            confidence=0.999
        )

        if not retval or inliers is None:
            logger.error("PnP RANSAC 计算失败！维度或几何可能发生严重退化")
            return None, None, None

        num_inliers = len(inliers)
        
        if num_inliers < 10:
            logger.warning(f"PnP 内点数极度匮乏 ({num_inliers} < 8)，为保全全局拓扑，放弃该帧注册")
            return None, None, None

        if num_inliers < 15:
            logger.warning(f"PnP 内点较少 ({num_inliers})，位姿可能不稳定。")

        inlier_idx = inliers.flatten()
        pts_3d_inliers = pts_3d[inlier_idx]
        pts_2d_inliers = pts_2d[inlier_idx]

        # Second pass: nonlinear refinement from the RANSAC pose.
        success, rvec_refined, tvec_refined = cv2.solvePnP(
            pts_3d_inliers, pts_2d_inliers, K, distCoeffs=None,
            rvec=rvec, tvec=tvec, 
            useExtrinsicGuess=True, 
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            R, _ = cv2.Rodrigues(rvec_refined)
            return R, tvec_refined, inliers
        else:
            logger.warning("PnP Refinement 优化不收敛，启用 RANSAC 原始结果保底。")
            R, _ = cv2.Rodrigues(rvec)
            return R, tvec, inliers
