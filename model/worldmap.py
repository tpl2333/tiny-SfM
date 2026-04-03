import cv2
import numpy as np
from pathlib import Path

from model.camera import Camera
from model.mappoint import Point
from model.frame import Frame
from model.edge import EdgeData

class Map:

    def __init__(self, camera:Camera):

        # 目前先单相机吧
        self._camera = camera
        # self.points = {point.idx: Point}
        self._points = {}
        # self.frames = {frame.idx: Frame}
        self._frames = {} 
        # registered_idx = set(registered_frame.idx)
        self._registered_ids = set()

        # self.view_graph = {frame.idx: {neighbor_frame.idx: obj EdgeData}}
        self._view_graph = {}

    # -------帧的创建与加载-------
    def create_frame(self, img_path):
        
        frame = Frame(img_path, self._camera)
        self._frames[frame.idx] = frame
            
    def load_frame_dir(self, img_dir):

        allowed_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        dir_path = Path(img_dir)
        try:
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix.lower() in allowed_suffixes:
                    frame = Frame(str(item),self._camera)
                    self._frames[frame.idx] = frame
                    self._view_graph[frame.idx] = {}
        except FileNotFoundError as e:
            print(f"[Map] Frame Directory Not Found!:{e}")

    # -------场景图的创建与检索-------

    def add_view_graph_edge(self, idx1, idx2, inlier_matches, inlier_ratio, model_type, score_f, score_h):

        edge_data = EdgeData(idx1, idx2, inlier_matches, inlier_ratio, model_type, score_f, score_h)

        if not edge_data.is_graph:
            return False

        if idx1 not in self._view_graph:
            self._view_graph[idx1] = {}
        if idx2 not in self._view_graph:
            self._view_graph[idx2] = {}

        self._view_graph[idx1][idx2] = edge_data
        self._view_graph[idx2][idx1] = edge_data
        
        return True
    
    def select_init_2_frames(self):
        pass

    def select_next_frames(self):
        pass


    # -------帧与地图点的注册、观测与检索------- 
    def register_frame(self, frame_idx, R, t):

        if frame_idx not in self._frames:
            raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")
        if frame_idx in self._registered_ids:
            print(f"[Map: Warning] Frame {frame_idx} has already registered, update the pose")

        frame = self._frames[frame_idx]
        frame.set_pose(R, t)
        self._registered_ids.add(frame.idx)
        frame.is_registered = True

    @property
    def unregistered_frames(self):
        # unregistered_frames = List[unregistered_frame.idx]
        return [fid for fid in self._frames if fid not in self._registered_ids]

    def create_point(self, position3d, color=None):

        return Point(position3d, color)
    
    def register_point(self, point:Point):

        if point.idx is not None:
            return point.idx 

        point.set_index() 
        self._points[point.idx] = point
        return point.idx
    
    def add_observation(self, point_idx, frame_idx, feature_idx):

        if point_idx not in self._points:
            raise KeyError(f"[Map] Point ID {point_idx} not in Map")
        if frame_idx not in self._frames:
            raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")

        point = self._points[point_idx]
        frame = self._frames[frame_idx]

        point.add_observation(frame_idx, feature_idx)
        frame.add_observation(feature_idx, point_idx)
        
    def get_point(self, point_idx):
        return self._points.get(point_idx)

    def get_frame(self, frame_idx):
        return self._frames.get(frame_idx)
    
    def all_frames(self):
        for fid in self._frames.keys():
            yield self._frames[fid]