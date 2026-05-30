import numpy as np
import logging
logger = logging.getLogger(__name__)

from management.viewgraph import ViewGraph


class FeatureTrack:
    """A multi-view feature track.

    observations stores pairs of (frame_idx, feature_idx). Once a track is
    triangulated, mappoint_idx links it to a Point in Map.
    """

    def __init__(self, track_idx):
        self.idx = track_idx 
        self.observations = []       
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
    """Build and query global feature tracks from the view graph."""

    def __init__(self):
        self._tracks = {}
        # Temporary union-find state used while building tracks.
        self._parent = {}
        # Persistent lookup from a 2D observation to its global track.
        self._feat_to_track = {}
        
        self._track_counter = 0

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

    def build_from_viewgraph(self, viewgraph:ViewGraph, threshold = 2):
        """Merge pairwise matches into global feature tracks."""
        logger.info("正在构建全局特征轨迹...")
        
        for idx1, idx2, edge_data in viewgraph.get_all_edges():
            for m in edge_data.matches:
                node1 = (idx1, m[0])
                node2 = (idx2, m[1])
                
                if node1 not in self._parent: self._parent[node1] = node1
                if node2 not in self._parent: self._parent[node2] = node2
                
                self._union(node1, node2)

        groups = {}
        for node in self._parent:
            root = self._find(node)
            groups.setdefault(root, []).append(node)

        self._tracks = {}
        self._feat_to_track = {}

        for root, obs_list in groups.items():
            if len(obs_list) < threshold:
                continue 
            # A valid track has at most one feature observation per frame.
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
        logger.info(f"构建完成，共生成 {len(self._tracks)} 条合法轨迹。")

    def _has_conflict(self, obs_list):
        """Return True if one frame contributes multiple features."""
        frame_ids = [o[0] for o in obs_list]
        return len(frame_ids) != len(set(frame_ids))
    
    def get_track_from_feat(self, frame_idx:int, feat_idx:int)->FeatureTrack:
        track_idx = self._feat_to_track.get((frame_idx, feat_idx))
        return self._tracks.get(track_idx)
    
    def get_track_from_idx(self, track_idx:int)->FeatureTrack:
        return self._tracks.get(track_idx)
    
    def reset_track_state(self, track_idx):

        track = self._tracks.get(track_idx)
        if track:
            track.mappoint_idx = None
            logger.debug(f"TrackManager: 轨迹 {track_idx} 已重置为未三角化状态")
    
    def classify_matches(self, frame1_idx, frame2_idx, inlier_matches=None):
        """Split matches into already-observed tracks and new triangulation tracks."""

        obs_tracks = []
        tri_tracks = []
        obs_matches = []
        tri_matches = []

        for m in inlier_matches:

            track = self.get_track_from_feat(frame1_idx, m[0])
            
            if track is None:
                continue
                
            if track.is_triangulated:
                obs_tracks.append(track.idx)
                obs_matches.append(m)
            else:
                tri_tracks.append(track.idx)
                tri_matches.append(m)

        obs_matches = np.array(obs_matches, dtype=np.int32).reshape(-1, 2)
        tri_matches = np.array(tri_matches, dtype=np.int32).reshape(-1, 2)

        return obs_tracks, obs_matches, tri_tracks, tri_matches
    
    def get_2d_3d_pairs(self, frame_idx):
        """Return feature indices and mapped 3D point ids for PnP."""
        feat_indices = []
        pt_indices = []

        for (f_idx, k_idx), track_idx in self._feat_to_track.items():
            if f_idx != frame_idx:
                continue
                
            track = self._tracks.get(track_idx)
            
            if track and track.is_triangulated:
                feat_indices.append(k_idx)
                pt_indices.append(track.mappoint_idx)

        return feat_indices, pt_indices
    
    def update_track_state(self, point_info: list, point_indices: list):
        """Bind newly created map point ids back to their feature tracks."""
        for info, p_idx in zip(point_info, point_indices):
            track_idx = info[0]
            track = self._tracks.get(track_idx)
            
            if track:
                track.connect_mappoint(p_idx)
            else:
                logger.error(f"Track {track_idx} not found during status update!")

