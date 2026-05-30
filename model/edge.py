import numpy as np

class EdgeData:
    """Verified pairwise relation between two frames in the view graph."""

    def __init__(self, inlier_matches, inlier_ratio, model_type, score_f, score_h):
        # matches[:, 0] indexes features in the lower-id/query frame.
        # matches[:, 1] indexes features in the higher-id/train frame.
        if isinstance(inlier_matches, list) and len(inlier_matches) > 0:
            self.matches = np.array(
                [[m.queryIdx, m.trainIdx] for m in inlier_matches], 
                dtype=np.int32
            )
        else:
            raise ValueError(f"[EdgeData]")

        self.num_inliers = self.matches.shape[0]
        self.inlier_ratio = inlier_ratio

        self.is_good = self.is_valid_for_graph()
        
        self.model_type = model_type
        self.score_f = score_f
        self.score_h = score_h

    @property
    def query_indices(self):
        """Feature indices from the first frame of the edge."""
        return self.matches[:, 0]

    @property
    def train_indices(self):
        """Feature indices from the second frame of the edge."""
        return self.matches[:, 1]
    
    def is_valid_for_graph(self):

        if self.num_inliers < 30:
            return False
        if self.inlier_ratio < 0.3:
            return False
        return True
