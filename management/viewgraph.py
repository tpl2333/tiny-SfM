import numpy as np
import logging
logger = logging.getLogger(__name__)


from model.edge import EdgeData

class ViewGraph:
    """Undirected graph of verified image-pair relations."""

    def __init__(self):
        # Edge keys are normalized as (min_frame_id, max_frame_id).
        self._edges = {}
        self._adjacency = {}
        self._registered_edges = set()

    def add_edge(self, id1, id2, edge_data:EdgeData):
        u, v = (id1, id2) if id1 < id2 else (id2, id1)
        self._edges[(u, v)] = edge_data
        
        self._adjacency.setdefault(u, set()).add(v)
        self._adjacency.setdefault(v, set()).add(u)

    def get_all_edges(self):
        """Iterate over all stored frame-pair edges."""
        for (id1, id2), edge_data in self._edges.items():
            yield id1, id2, edge_data

    def get_connected_frames(self, frame_id):
        """Return all frames connected to the given frame."""
        return self._adjacency.get(frame_id, set())

    def get_edge(self, frame_idx1:int, frame_idx2:int) -> EdgeData:
        """Return the edge data for a frame pair if it exists."""
        u, v = (frame_idx1, frame_idx2) if frame_idx1 < frame_idx2 else (frame_idx2, frame_idx1)
        return self._edges.get((u,v))




