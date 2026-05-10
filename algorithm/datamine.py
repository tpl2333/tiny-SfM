import numpy as np
import logging
logger = logging.getLogger(__name__)

from management.viewgraph import ViewGraph
from management.worldmap import Map
from management.trackmanager import TrackManager


class DataMiner:
    
    def find_best_seed(self, viewgraph: ViewGraph, worldmap: Map):
        best_score = -1
        best_pair = None
        
        for id1, id2, edge in viewgraph.get_all_edges():
            if edge.model_type != 'F' or not edge.is_good:
                continue
                
            # --- 维度 1: 匹配数量 (压制相邻帧的暴利) ---
            # 超过 100 个点后，数量带来的边际收益递减
            num_score = np.log10(edge.num_inliers) 
            
            # --- 维度 2: 几何质量 (GRIC 比例) ---
            gric_ratio = edge.score_h / edge.score_f 
            
            # --- 维度 3: 空间分布 (新增) ---
            frame1 = worldmap.get_frame(id1)
            spread_score = self.calculate_spatial_spread(frame1, edge.query_indices)
            
            # 综合初步得分
            current_score = num_score * gric_ratio * spread_score
            
            # --- 维度 4: 强制视差阈值 (只有得分领先的才跑这个，省资源) ---
            # if current_score > threshold:
            #     # 尝试用 mvgsolver 算一下位姿和初步视差
            #     # 如果 median_parallax < 2.0: continue
            #     pass

            if current_score > best_score:
                best_score = current_score
                best_pair = (id1, id2, edge)

        return best_pair

    def calculate_spatial_spread(self, frame, feature_indices):
        """ 计算内点在图像中的覆盖率 (0.0 ~ 1.0) """
        grid_size = 8
        grid = np.zeros((grid_size, grid_size))
        h = frame.height
        w = frame.weight 
        
        for idx in feature_indices:
            pt = frame.kps[idx].pt
            # 映射到网格坐标
            gx = int(pt[0] * grid_size / w)
            gy = int(pt[1] * grid_size / h)
            grid[min(gy, grid_size-1), min(gx, grid_size-1)] = 1
            
        return np.sum(grid) / (grid_size * grid_size)
    
    def find_next_best_frame(self, worldmap:Map, viewgraph:ViewGraph, trackmanager:TrackManager):
        """
        寻找下一个最适合注册的帧
        """
        registered_ids = worldmap.registered_frame_set
        unregistered_ids = worldmap.unregistered_frame_set
        
        best_frame_idx = None
        max_correspondences = 0
        
        # 评分字典，用于调试记录
        scores = {}

        for un_idx in unregistered_ids:
            # 1. 检查邻居：只看那些与已注册帧有连接的候选者
            neighbors = viewgraph.get_connected_frames(un_idx)
            if not (neighbors & registered_ids):
                continue
            
            # 2. 核心：通过 TrackManager 统计 2D-3D 对应关系
            # 统计候选帧有多少个特征点所属的 Track 已经有了 3D 点
            corr_count = 0
            frame_obj = worldmap.get_frame(un_idx)
            num_features = len(frame_obj.kps)
            
            for feat_idx in range(num_features):
                track = trackmanager.get_track_from_feat(un_idx, feat_idx)
                if track and track.is_triangulated:
                    corr_count += 1
            
            scores[un_idx] = corr_count
            
            # 3. 更新最大值
            if corr_count > max_correspondences:
                max_correspondences = corr_count
                best_frame_idx = un_idx

        # 阈值检查：如果最多的也只有不到 30 个点，建议停止重建或报错
        if best_frame_idx is None:
            logger.info(f"没有选定下一个候选帧")
        elif max_correspondences < 15:
            logger.warning(f"候选帧 {best_frame_idx} 中最大关联数仅为 {max_correspondences}，重建质量可能下降")

        return best_frame_idx, max_correspondences