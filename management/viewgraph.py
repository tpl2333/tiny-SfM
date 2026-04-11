import numpy as np

from model.edge import EdgeData

class ViewGraph:
    # ----------图的初始化方法---------
    def __init__(self):
        # 存储边：{(frame_id1, frame_id2): EdgeData}  其中 id1<id2
        self._edges = {}
        # 邻接表: {frame_id: set(neighbor_ids)}
        self._adjacency = {}
        # 注册表：set{frame_id1,....}
        self._registered_edges = set()

    def add_edge(self, id1, id2, edge_data:EdgeData):
        u, v = (id1, id2) if id1 < id2 else (id2, id1)
        self._edges[(u, v)] = edge_data
        
        self._adjacency.setdefault(u, set()).add(v)
        self._adjacency.setdefault(v, set()).add(u)

    # ---------图的各种查询方法---------
    def get_all_edges(self):
        """遍历所有的边，用于构建 Feature Tracks"""
        for (id1, id2), edge_data in self._edges.items():
            yield id1, id2, edge_data

    def get_connected_frames(self, frame_id):
        """获取与某帧相连的所有帧，用于扩展地图"""
        return self._adjacency.get(frame_id, set())

    def get_edge(self, frame_idx1:int, frame_idx2:int) -> EdgeData:
        """获取两张图之间匹配信息 (edgedata)"""
        u, v = (frame_idx1, frame_idx2) if frame_idx1 < frame_idx2 else (frame_idx2, frame_idx1)
        return self._edges.get((u,v))

    # ---------图的状态更新方法---------
    def register_frame(self, frame_id):

        self._registered_edges.add(frame_id)


