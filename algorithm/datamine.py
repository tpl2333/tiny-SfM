import logging
logger = logging.getLogger(__name__)

from management.viewgraph import ViewGraph
from management.worldmap import Map
from management.trackmanager import TrackManager


class DataMiner:
    
    def find_best_seed(self, viewgraph:ViewGraph):
        """
        从 ViewGraph 中搜索最适合作为起点的边
        """
        best_score = -1
        best_pair = None
        
        # 遍历 ViewGraph 中所有的边
        for id1, id2, edge in viewgraph.get_all_edges():
            # 1. 基础过滤：必须是通过 F 矩阵校验的边
            if edge.model_type != 'F' or not edge.is_good:
                continue
                
            # 2. 计算初始化权重分数
            # 核心思路：内点越多越好，且 F 分数要比 H 分数优势大（代表视差足）
            # 注意：GRIC 分数越小表示模型拟合越好
            gric_ratio = edge.score_h / edge.score_f 
            current_score = edge.num_inliers * gric_ratio
            
            if current_score > best_score:
                best_score = current_score
                best_pair = (id1, id2, edge)
        
        if best_pair is None:
            raise RuntimeError("无法在 ViewGraph 中找到合适的初始化种子对！")
            
        return best_pair
    
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