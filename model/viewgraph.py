import numpy as np

class EdgeData:
    def __init__(self, idx1, idx2, inlier_matches, inlier_ratio, model_type, score_f, score_h):
        self.idx1 = idx1
        self.idx2 = idx2

        if isinstance(inlier_matches, list) and len(inlier_matches) > 0:
            self.matches = np.array(
                [[m.queryIdx, m.trainIdx] for m in inlier_matches], 
                dtype=np.int32
            )
        else:
            self.matches = np.array(inlier_matches, dtype=np.int32).reshape(-1, 2)

        self.num_inliers = self.matches.shape[0]
        self.inlier_ratio = inlier_ratio

        self.is_graph = self.is_valid_for_graph()
        self.is_init = self.is_suitable_for_seed()
        
        self.model_type = model_type  # 'F' 或 'H'
        self.score_f = score_f
        self.score_h = score_h

    @property
    def query_indices(self):
        """返回第一张图的所有内点索引"""
        return self.matches[:, 0]

    @property
    def train_indices(self):
        """返回第二张图的所有内点索引"""
        return self.matches[:, 1]

    def get_other_id(self, my_idx):
        return self.idx2 if my_idx == self.idx1 else self.idx1
    
    def is_valid_for_graph(self):

        if self.num_inliers < 30:
            return False
        if self.inlier_ratio < 0.3:
            return False
        return True
        
    def is_suitable_for_seed(self):

        if not self.is_valid_for_graph():
            return False
        if self.model_type != "F":
            return False  
        if self.score_f > self.score_h:
            return False
        return True