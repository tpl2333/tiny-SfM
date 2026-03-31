import cv2
import numpy as np
from pathlib import Path

from model.camera import Camera
from model.mappoint import Point
from model.frame import Frame

class Map:

    @property
    def unregistered_frames(self):
        # unregistered_frames = List[unregistered_frame.idx]
        return [fid for fid in self._frames if fid not in self._registered_ids]
    
    def __init__(self, camera:Camera):

        # 目前先单相机吧
        self._camera = camera
        # self.points = {point.idx: Point}
        self._points = {}
        # self.frames = {frame.idx: Frame}
        self._frames = {} 
        # registered_idx = set(registered_frame.idx)
        self._registered_ids = set()
    
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
        except FileNotFoundError as e:
            print(f"[Map] Frame Directory Not Found!:{e}")

    def register_frame(self, frame_idx, R, t):

        if frame_idx not in self._frames:
            raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")
        if frame_idx in self._registered_ids:
            print(f"[Map: Warning] Frame {frame_idx} has already registered, update the pose")

        frame = self._frames[frame_idx]
        frame.set_pose(R, t)
        self._registered_ids.add(frame.idx)
        frame.is_registered = True

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
    

        
