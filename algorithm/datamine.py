from management.viewgraph import ViewGraph

class DataMiner:
    
    def find_best_seed(self, view_graph:ViewGraph):
        """
        从 ViewGraph 中搜索最适合作为起点的边
        """
        best_score = -1
        best_pair = None
        
        # 遍历 ViewGraph 中所有的边
        for id1, id2, edge in view_graph.get_all_edges():
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
    
    def find_next_frame(self):
        pass