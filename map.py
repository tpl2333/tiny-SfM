import cv2
import numpy as np
from camera import Camera
from mappoint import Point
from frame import Frame

class Map:
    def __init__(self):
        
        self.map_points = [] # 存储所有的 MapPoint 对象实例
        self.keyframes = []  # 存储所有的关键帧 Frame 实例

    def add_point(self, mp):
        self.map_points.append(mp)
    
    def add_frame(self, frame):
        self.keyframes.append(frame)
        
    def get_all_points_as_array(self):
        # 专门为了 Open3D 可视化提供的方法
        # 遍历对象列表，提取 position 拼成 numpy 数组
        return np.array([mp.position for mp in self.map_points])