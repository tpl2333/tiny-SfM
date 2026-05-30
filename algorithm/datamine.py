import numpy as np
import logging
logger = logging.getLogger(__name__)

from management.viewgraph import ViewGraph
from management.worldmap import Map
from management.trackmanager import TrackManager


class DataMiner:
    
    def find_best_seed(self, viewgraph: ViewGraph, worldmap: Map):
        """Select an initial image pair from the view graph.

        The seed pair should have enough verified matches, prefer a general
        3D fundamental-matrix relation, and cover a broad area of the image.
        """
        best_score = -1
        best_pair = None
        
        for id1, id2, edge in viewgraph.get_all_edges():
            if edge.model_type != 'F' or not edge.is_good:
                continue
                
            # Use logarithmic match count so dense but redundant pairs do not dominate.
            num_score = np.log10(edge.num_inliers) 
            
            # Larger H/F GRIC ratio means the fundamental model is more favored.
            gric_ratio = edge.score_h / edge.score_f 
            
            # Prefer features spread across the image instead of clustered texture.
            frame1 = worldmap.get_frame(id1)
            spread_score = self.calculate_spatial_spread(frame1, edge.query_indices)
            
            current_score = num_score * gric_ratio * spread_score

            if current_score > best_score:
                best_score = current_score
                best_pair = (id1, id2, edge)

        return best_pair

    def calculate_spatial_spread(self, frame, feature_indices):
        """Estimate how widely the selected features cover the image."""
        grid_size = 8
        grid = np.zeros((grid_size, grid_size))
        h = frame.height
        w = frame.weight 
        
        for idx in feature_indices:
            pt = frame.kps[idx].pt
            gx = int(pt[0] * grid_size / w)
            gy = int(pt[1] * grid_size / h)
            grid[min(gy, grid_size-1), min(gx, grid_size-1)] = 1
            
        return np.sum(grid) / (grid_size * grid_size)
    
    def find_next_best_frame(self, worldmap: Map, viewgraph: ViewGraph, trackmanager: TrackManager):
        """Choose the next frame for PnP registration.

        A good candidate observes enough existing 3D tracks and those
        observations should be spatially distributed in the image.
        """
        registered_ids = worldmap.registered_frame_set
        unregistered_ids = worldmap.unregistered_frame_set
        
        best_frame_idx = None
        best_score = -1
        max_correspondences = 0
        
        for un_idx in unregistered_ids:
            # Only frames connected to the current reconstruction can be registered.
            neighbors = viewgraph.get_connected_frames(un_idx)
            if not (neighbors & registered_ids):
                continue
            
            corr_count = 0
            feature_indices_with_3d = []
            frame_obj = worldmap.get_frame(un_idx)
            num_features = len(frame_obj.kps)
            
            for feat_idx in range(num_features):
                track = trackmanager.get_track_from_feat(un_idx, feat_idx)
                if track and track.is_triangulated:
                    corr_count += 1
                    feature_indices_with_3d.append(feat_idx)
            
            # Keep a small margin above the minimal PnP sample size for RANSAC.
            if corr_count < 12:
                continue
            
            spread_score = self.calculate_spatial_spread(frame_obj, feature_indices_with_3d)
            current_score = np.log10(corr_count) * spread_score
            
            if current_score > best_score:
                best_score = current_score
                best_frame_idx = un_idx
                max_correspondences = corr_count

        if best_frame_idx is None:
            logger.info("没有选定下一个候选帧")
        elif max_correspondences < 20:
            logger.warning(f"候选帧 {best_frame_idx} 仅有 {max_correspondences} 个 2D-3D 关联，但空间分布评分良好，尝试注册。")

        return best_frame_idx, max_correspondences
