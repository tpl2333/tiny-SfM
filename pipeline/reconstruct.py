import cv2
import numpy as np
import open3d as o3d
from algorithm.match import FeatureMatcher
from algorithm.errors import *
from management.worldmap import Map

class Reconstructor:
    def __init__(self, worldmap:Map, matcher:FeatureMatcher, frame_dir):

        self.map = worldmap
        self.matcher = matcher
        self.frame_dir = frame_dir

        # 参数什么的写这里
        self.parameter = None

    # -------顶层运行层(temporal)-------
    def run(self):
        """
        多视图重建主循环：从初始化开始，逐帧注册并扩展地图。
        注：当前版本不包含 BA 优化。
        """
        print("-" * 30)
        print("[Reconstructor] 启动重建流水线...")
        print(f"[Reconstructor] 图像路径: {self.frame_dir}")
        print("-" * 30)

        # 1. 初始化阶段 (建立初始两帧的相对位姿与初始点云)
        try:
            # initialize_worldmap 内部已经处理了 load_frame_dir 和前两帧的匹配与三角化
            self.initialize_worldmap(self.frame_dir)
            print(f"[Reconstructor] 初始化成功。")
            print(f"                已注册帧: {list(self.map._registered_ids)}")
            print(f"                当前地图点数: {len(self.map._points)}")
        except Exception as e:
            print(f"[Reconstructor] 初始化失败: {e}")
            return

        # 2. 增量注册阶段 (处理剩余的帧)
        # 只要还有未注册的帧，就继续推进
        while len(self.map.unregistered_frames) > 0:
            
            # 获取下一帧的 ID (用于打印 log)
            next_frame_id = self.map.unregistered_frames[0]
            
            print(f"\n[Reconstructor] 正在尝试注册帧 ID: {next_frame_id} ...")
            
            # 调用你刚才完成的 add_next_frame
            # 该函数内部会执行：提取特征 -> 2D-3D 匹配 -> PnP 定位 -> 三角化新点
            success = self.add_next_frame()

            if success:
                print(f"[Reconstructor] 帧 {next_frame_id} 注册完成。")
                print(f"当前总地图点数: {len(self.map._points)}")
            else:
                # 如果某一帧失败了（比如匹配点太少），为了防止死循环，
                # 我们需要将其强制从待处理列表中移除（或者跳过）
                print(f"[Reconstructor] 警告: 帧 {next_frame_id} 注册失败，尝试跳过。")
                
                # 简单处理：如果 add_next_frame 失败且没被标记注册，
                # 我们手动把它放进已注册列表（即使位姿是错的或空的），或者直接从待处理里删掉
                # 这里为了严谨，我们可以暂时把这帧“冷冻”，在 self.map 中标记它处理过但失败了
                self.map._registered_ids.add(next_frame_id) 

        print("\n" + "-" * 30)
        print("[Reconstructor] 重建结束。")
        print(f"最终结果: 共注册 {len(self.map._registered_ids)} 帧, 生成 {len(self.map._points)} 个地图点。")
        print("-" * 30)

    # -------任务编排层(temporal)-------
    def initialize_worldmap(self, frame_dir):

        # 1. 初始化map
        self.map.load_frame_dir(frame_dir)

        # 2. 选取初始化帧对 视差匹配对等等.....
        # 这里假装挑出来了两帧，其实是第一第二帧
        frame1_idx = self.map.unregistered_frames[0]
        frame2_idx = self.map.unregistered_frames[1]

        # 3. 提取初始化帧对的图像特征
        self._extract_single_frame(frame1_idx)
        self._extract_single_frame(frame2_idx)
        print("[Reconstructor]: initial feature extracted!")

        # 4. 匹配初始化帧对的特征并评估模型情况
        _, inlier_matches,_ ,model_type, _, _ = self._match_2d_pair(frame1_idx, frame2_idx)
        if model_type != "F":
            raise ValueError(f"[Reconstructor]: model type is not general 3D")
        print(f"[Reconstructor]: inital feature matched and vertificated by GRIC, model is {model_type}")

        # 5. 根据本质矩阵分解注册最初的两帧
        D_inlier_matches = self._register_initial_frames(frame1_idx, frame2_idx, inlier_matches)
        print(f"[Reconstructor]: get {len(D_inlier_matches)} D_matches!")

        # 6. 根据D_inlier进行三角化，并使用几何审核得到注册点候选
        candidates = self._triangulate_between_frames(frame1_idx, frame2_idx, D_inlier_matches)
        print(f"[Reconstructor]: get {len(candidates)} candidated mappoint for register!")

        # 7. 使用统计审核并注册得到最终的地图点
        new_add_count = self._register_tri_candidates(frame1_idx,frame2_idx, candidates)
        print(f"[Reconstructor]: register new {new_add_count} mappoints!")

    def add_next_frame(self):

        # 1. 检查是否有待处理的帧
        unregistered = self.map.unregistered_frames
        if not unregistered:
            print("[Reconstructor] No more frames to add.")
            return False
            
        new_frame_idx = unregistered[0]
        # 假设我们总是相对于上一帧（最近注册的那一帧）进行匹配
        # 在更复杂的系统中，通常会维护一个“关键帧”列表进行匹配
        last_frame_idx = list(self.map._registered_ids)[-1]

        print(f"[Reconstructor] Processing Frame {new_frame_idx} (ref: {last_frame_idx})...")

        # 2. 提取新帧特征
        self._extract_single_frame(new_frame_idx)

        # 3. 与上一帧进行 2D 匹配
        # matrix 在这里可能是 F 或 H，在 PnP 流程中我们主要关注 F_inlier_matches
        _, inlier_matches,_ ,model_type, _, _ = self._match_2d_pair(last_frame_idx, new_frame_idx)
        if model_type != "F":
            raise ValueError(f"[Reconstructor]: model type is not general 3D")

        # 4. 寻找 2D-3D 对应关系
        # 这一步是关键：找出新帧中哪些特征点对应的 3D 点已经在地图中了
        pts_3d, pts_2d, pts_3d_ids, pts_2d_ids = self._find_2d_3d_correspondences(
            last_frame_idx, new_frame_idx, inlier_matches
        )

        if len(pts_3d) < 7: # PnP 至少需要 4-6 对点，这里设 7 对保证 RANSAC 鲁棒性
            print(f"[Reconstructor] Too few 2D-3D correspondences ({len(pts_3d)})!")
            return False

        # 5. PnP 注册位姿
        # 此函数内部会调用 cv2.solvePnPRansac 并执行 self.map.register_frame
        try:
            self._register_frame_pnp(new_frame_idx, pts_3d, pts_2d, pts_3d_ids, pts_2d_ids)
        except ValueError as e:
            print(f"[Reconstructor] PnP registration failed: {e}")
            return False

        # 6. 三角化新点 (Map Expansion)
        # 上一步 PnP 只处理了已有的点，这一步将匹配对中“还没成点”的部分转为 3D
        candidates = self._triangulate_between_frames(last_frame_idx, new_frame_idx, inlier_matches)

        # 7. 注册新生成的地图点并建立观测关系
        new_points_count = self._register_tri_candidates(last_frame_idx, new_frame_idx, candidates)
        print(f"[Reconstructor] Frame {new_frame_idx} registered. Added {new_points_count} new points.")
        return True 
    # -------顶层运行层(unordered)-------

    # -------任务编排层(unordered)-------
    def init_worldmap_and_viewgraph(self, frame_dir):

        self.map.load_frame_dir(frame_dir)

        self._extract_all_frames()

        self._build_view_graph()

        self._get
    
    def init_pose(self):
        pass

    def add_next_frame(self):
        pass

    # -------内部逻辑层-------
    def _extract_all_frames(self):
        """extract all frames' features
        """
        frames = list(self.map.all_frames())
        if not frames:
            raise ValueError(f"[Reconstructor]: extract all frames failed, world map has no frame!")
        for frame in frames:
            self.matcher.extract(frame)

    def _extract_single_frame(self, frame_idx):
        """ extract features from frame_idx
        Args:
            frame_idx (int)
        """
        frame = self.map.get_frame(frame_idx)
        self.matcher.extract(frame)

    def _match_2d_pair(self, frame1_idx, frame2_idx):
        """ match features of frame1 and frame2 

        Args:
            frame1_idx (int)
            frame2_idx (int)

        Returns:
            matrix: model matrix (H/F)
            F_inlier_matches: vertificated matches by F/H
            model_type: "F" fundamental/"H" homography
        """

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
        """ compute inital pose, register inital frames and return D_inlier

        Args:
            frame1_idx
            frame2_idx
            F_inlier_matches
        
        Returns:
            D_inlier_matches: vertificated matches by depth
        """

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
        """_summary_

        Args:
            frame1_idx: old_frame
            frame2_idx: new_frame
            D_inlier_matches: _description_

        Raises:
            TriangulateError

        Returns:
            candidates: List[(Point,feature1_idx,feature2_idx)]
        """

        frame1 = self.map.get_frame(frame1_idx)
        frame2 = self.map.get_frame(frame2_idx)

        P1 = frame1.get_proj_matrix()
        P2 = frame2.get_proj_matrix()

        tri_matches = []

        # 筛选三角化匹配点
        # 两帧的两个特征 均关联 mappoint 无事发生（point_id 不同可能需要合并）
        # 两帧的一个特征 关联 mappoint   另外帧的特征添加关联
        # 两帧均无关联 mappoint         三角化候选点
        for m in inlier_matches:

            id1, id2 = m.queryIdx, m.trainIdx 
            
            p1 = frame1.feature_2_point.get(id1)
            p2 = frame2.feature_2_point.get(id2)

            if p1 is not None and p2 is None:
                self.map.add_observation(p1, frame2_idx, id2)
            elif p1 is None and p2 is not None:
                self.map.add_observation(p2, frame1_idx, id1)
            elif p1 is not None and p2 is not None:
                # 先不管，跳过可能的merge
                continue
            else:
                tri_matches.append(m)
            
        pts1 = np.float32([frame1.kps[m.queryIdx].pt for m in tri_matches]).reshape(-1, 2).T
        pts2 = np.float32([frame2.kps[m.trainIdx].pt for m in tri_matches]).reshape(-1, 2).T

        points4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

        if points4d is None:
            raise TriangulateError(f"[reconstructor]: triangulation bewteen frame {frame1_idx} and {frame2_idx} failed")

        xyz = points4d[:3, :] #[3, N]
        w = points4d[3:, :]   #[1, N]
        points_normalized = (xyz / w).T #归一化并转置 [N, 3]

        
        o1 = frame1.get_center().flatten()
        o2 = frame2.get_center().flatten()
        R1, t1 = frame1.R, frame1.t 
        R2, t2 = frame2.R, frame2.t
        MIN_PARALLAX_DEG = 1.0
        max_cos_threshold = np.cos(np.deg2rad(MIN_PARALLAX_DEG))

        candidates = []
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
            f1_feat_idx = match.queryIdx
            f2_feat_idx = match.trainIdx

            # 获取颜色
            u, v = map(int, frame1.kps[f1_feat_idx].pt) 
            bgr = frame1.get_color(u,v) 
            rgb = bgr[::-1] / 255.0

            new_point = self.map.create_point(x,color=rgb)
            candidates.append((new_point, f1_feat_idx, f2_feat_idx))    

        return candidates
    
    def _find_2d_3d_correspondences(self, last_frame_idx: int, new_frame_idx: int, F_inlier_matches: list[cv2.DMatch]):
        """ find correspondences of keypoints in new frame and observed mappoints in last frame 

        Args:
            last_frame_idx (int)
            new_frame_idx (int)
            F_inlier_matches (list[cv2.DMatch])

        Returns:
            pts_3d (array)
            pts_2d (array)
            pts_3d_ids (list[int])
            pts_2d_ids (list[int])
        """
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
        """ register frame and points using PnP_iterative

        Args:
            new_frame_idx (int)
            pts_3d (np.ndarray)
            pts_2d (np.ndarray)
            pts_3d_ids (list[int])
            pts_2d_ids (list[int])

        Raises:
            ValueError: _description_

        Returns:
            PnP_inlier_matches: 
        """

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
            # 额外引入统计审核规则
            # 注册点到全局地图
            point_idx = self.map.register_point(point)
            
            # 建立双向观测关系
            self.map.add_observation(point_idx, last_frame_idx, feat1_idx)
            self.map.add_observation(point_idx, new_frame_idx, feat2_idx)
            new_points_count += 1
        
        return new_points_count
        


















    
        


