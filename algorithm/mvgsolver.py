import cv2
import numpy as np

from model.frame import Frame
from model.edge import EdgeData

from management.viewgraph import ViewGraph
from management.trackmanager import TrackManager
from management.worldmap import Map

from algorithm.errors import *

class MvgSolver:
    
    def get_initial_pose(self, frame1:Frame, frame2:Frame, edge:EdgeData):
        """ 计算本质矩阵获取初始位姿

        Args:
            frame1 (Frame): _description_
            frame2 (Frame): _description_
            edge (EdgeData): _description_
            view_graph (ViewGraph): _description_

        Raises:
            ValueError: _description_

        Returns:
            R: Matlike 旋转矩阵 W2C
            t: Matlike 平移向量 W2C
            D_inlier_matches: 经过手性检测后的内点，满足深度测试
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

    def triangulate(self, frame1:Frame, frame2:Frame, tri_tracks:list[int], tri_matches:np.ndarray):
        """ 三角化两帧之间的对应点

        Args:
            frame1 (Frame): _description_
            frame2 (Frame): _description_
            tri_tracks (list[int]): _description_
            tri_matches (np.ndarray): _description_

        Raises:
            TriangulateError: _description_

        Returns:
            point_info (list[tuple]): [(track_idx, position3d, color)]
        """
        if len(tri_matches) == 0:
            return []

        P1 = frame1.get_proj_matrix()
        P2 = frame2.get_proj_matrix()
            
        pts1 = np.float32([frame1.kps[m].pt for m in tri_matches[:,0]]).reshape(-1, 2).T
        pts2 = np.float32([frame2.kps[m].pt for m in tri_matches[:,1]]).reshape(-1, 2).T

        points4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

        if points4d is None:
            raise TriangulateError(f"[reconstructor]: triangulation bewteen frame {frame1.idx} and {frame2.idx} failed")

        xyz = points4d[:3, :] #[3, N]
        w = points4d[3:, :]   #[1, N]
        points_normalized = (xyz / w).T #归一化并转置 [N, 3]

        o1 = frame1.get_center().flatten()
        o2 = frame2.get_center().flatten()
        R1, t1 = frame1.R, frame1.t 
        R2, t2 = frame2.R, frame2.t
        MIN_PARALLAX_DEG = 1.0
        max_cos_threshold = np.cos(np.deg2rad(MIN_PARALLAX_DEG))

        point_info = []
        for i, x in enumerate(points_normalized):

            # 深度检测 
            p_c1 = R1 @ x.reshape(3,1) + t1
            p_c2 = R2 @ x.reshape(3,1) + t2
            if p_c1[2] <= 0 or p_c2[2] <= 0:
                continue

            # 视差角检测 
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

            # 获取颜色
            u, v = map(int, frame1.kps[f1_feat_idx].pt) 
            bgr = frame1.get_color(u,v) 
            rgb = bgr[::-1] / 255.0

            point_info.append((track_idx, x, rgb ))    

        return point_info