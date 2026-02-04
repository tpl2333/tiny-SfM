import cv2
import numpy as np
from model.camera import Camera
from model.mappoint import Point
from model.frame import Frame

class Map:
    def __init__(self):
        
        self.points = {} # 存储所有的 MapPoint 对象实例
        self.frames = {}  # 存储所有的关键帧 Frame 实例

    def add_point(self, mp:Point):
        if mp.idx not in self.points:
            self.points[mp.idx] = mp
    
    def add_frame(self, frame:Frame):
        if frame.idx not in self.frames:
            self.frames[frame.idx] = frame
        frame.is_registered = True
    
    def get_point(self, mp_idx):
        return self.points.get(mp_idx)

    def get_frame(self, frame_idx):
        return self.frames.get(frame_idx)
    
    def remove_point(self, mp_idx):
        if mp_idx in self.points:
            del self.points[mp_idx]