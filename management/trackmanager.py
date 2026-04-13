import numpy as np
import logging
logger = logging.getLogger(__name__)

from management.viewgraph import ViewGraph


class FeatureTrack:
    def __init__(self, track_idx):
        self.idx = track_idx 
        # 观测列表: [(frame_idx, feature_idx), ...]
        self.observations = []       
        # 关联的 3D 地图点 ID，None 表示无关联
        self.mappoint_idx = None

    @property
    def is_triangulated(self):
        return self.mappoint_idx is not None
    
    def connect_mappoint(self, point_idx):
        self.mappoint_idx = point_idx

    def add_observation(self, frame_idx, feat_idx):
        self.observations.append((frame_idx, feat_idx))

    def __len__(self):
        return len(self.observations)
    
class TrackManager:
    def __init__(self):
        # 特征轨迹: {track_idx: FeatureTrack}
        self._tracks = {}
        # 并查集维护字典: {(frame_idx, feat_idx): parent_node}
        self._parent = {}
        # 查询轨迹字典：{(frame_idx, feat_idx): track_idx}
        # 不能简单认为可以归类到并查集维护字典中。
        self._feat_to_track = {}
        
        self._track_counter = 0

    # ---------并查集基础方法---------
    def _find(self, node):
        if self._parent[node] == node:
            return node
        self._parent[node] = self._find(self._parent[node])
        return self._parent[node]

    def _union(self, node1, node2):
        root1 = self._find(node1)
        root2 = self._find(node2)
        if root1 != root2:
            self._parent[root1] = root2

    # ---------初始化方法---------
    def build_from_viewgraph(self, viewgraph:ViewGraph):
        """
        从 ViewGraph 中提取所有 EdgeData，串联成 Tracks
        """
        print("[TrackManager] 正在构建全局特征轨迹...")
        
        # 1. 遍历 ViewGraph 中所有的边
        for idx1, idx2, edge_data in viewgraph.get_all_edges():
            for m in edge_data.matches:
                node1 = (idx1, m[0]) # (query_frame_idx, query_feat_idx)
                node2 = (idx2, m[1]) # (train_frame_idx, train_feat_idx)
                
                if node1 not in self._parent: self._parent[node1] = node1
                if node2 not in self._parent: self._parent[node2] = node2
                
                self._union(node1, node2)

        # 2. 将并查集的分组结果转换为 FeatureTrack 对象
        # group = {root_node: [root_node, node1, node2, ...]}
        # node = (frame_idx, feat_idx)
        groups = {}
        for node in self._parent:
            root = self._find(node)
            groups.setdefault(root, []).append(node)

        # 3. 过滤并存储
        self._tracks = {}
        self._feat_to_track = {}

        for root, obs_list in groups.items():
            # 孤立点不构成轨迹
            if len(obs_list) < 2:
                continue 
            # 冲突检查：同一个轨迹在同一帧内不能有多个点
            if self._has_conflict(obs_list):
                continue

            track_idx = self._track_counter
            new_track = FeatureTrack(track_idx)
            new_track.observations = obs_list
            self._tracks[track_idx] = new_track

            for node in obs_list:
                self._feat_to_track[node] = track_idx

            self._track_counter += 1
        
        self._parent.clear()
        print(f"[TrackManager] 构建完成，共生成 {len(self._tracks)} 条合法轨迹。")

    def _has_conflict(self, obs_list):
        """检查是否有同一帧贡献了多个特征点"""
        frame_ids = [o[0] for o in obs_list]
        return len(frame_ids) != len(set(frame_ids))
    
    # ---------轨迹的查询，分类与状态更改---------
    def get_track(self, frame_idx:int, feat_idx:int)->FeatureTrack:
        track_idx = self._feat_to_track.get((frame_idx, feat_idx))
        return self._tracks.get(track_idx)
    
    def classify_matches(self, frame1_idx, frame2_idx, inlier_matches=None):
        """ 
        筛选出匹配对所属轨迹中未三角化的匹配对，避免重复三角化

        Args:
            frame1_idx (_type_): _description_
            frame2_idx (_type_): _description_
            inlier_matches (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """

        obs_tracks = []
        tri_tracks = []
        obs_matches = []
        tri_matches = []

        for m in inlier_matches:

            track = self.get_track(frame1_idx, m[0])
            
            if track is None:
                continue
                
            if track.is_triangulated:
                # 状态 A: 该轨迹已经有了地图点，无需重复添加观测。
                obs_tracks.append(track.idx)
                obs_matches.append(m)
            else:
                # 状态 B: 该轨迹还没有地图点，这是三角化的黄金机会。
                tri_tracks.append(track.idx)
                tri_matches.append(m)

        obs_matches = np.array(obs_matches, dtype=np.int32)
        tri_matches = np.array(tri_matches, dtype=np.int32)

        return obs_tracks, obs_matches, tri_tracks, tri_matches
    
    def get_2d_3d_pairs(self, frame_idx):
        """        
        获取某一帧中所有具有对应 3D 地图点的特征索引。
        用于 PnP 位姿解算。

        Args:
            frame_idx (int): 帧索引

        Returns:
            feature_indices: List[int] 该帧特征点的索引
            mappoint_indices: List[int] 对应的 3D 地图点 ID
        """
        feat_indices = []
        pt_indices = []

        # 1. 遍历该帧在 trackmanager 记录中的所有特征点
        for (f_idx, k_idx), track_idx in self._feat_to_track.items():
            if f_idx != frame_idx:
                continue
                
            track = self._tracks.get(track_idx)
            
            # 2. 如果该轨迹已经三角化了，这就是一个 PnP 约束！
            if track and track.is_triangulated:
                feat_indices.append(k_idx)
                pt_indices.append(track.mappoint_idx)

        return feat_indices, pt_indices
    
    def update_track_state(self, point_info: list, point_indices: list):
        """
        将地图生成的 point_idx 绑定回对应的 FeatureTrack
        point_info: [(track_idx, x, color), ...]
        point_indices: [p_idx1, p_idx2, ...]
        """
        # 使用 zip 同时迭代两个列表，防止索引越界
        for info, p_idx in zip(point_info, point_indices):
            track_idx = info[0]
            track = self._tracks.get(track_idx)
            
            if track:
                track.connect_mappoint(p_idx)
            else:
                logger.error(f"Track {track_idx} not found during status update!")

