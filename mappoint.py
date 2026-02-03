from __future__ import annotations
import cv2
import numpy as np
from camera import Camera
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from frame import Frame

class Point:

    _counter = 0

    def __init__(self, point):

        self.idx = Point._counter
        Point._counter += 1

        self.position = point
        self.color = None

        # 该3D点可以被观察到的图像以及对应的关键点 
        # 字典键值对： Frame.idx:index (index来自Frame.kps[index])
        self.observations = {}
    
    def add_observation(self, frame:Frame, feature_idx):

        if frame.idx not in self.observations:
            self.observations[frame.idx] = feature_idx
            # 可以在这里增加计数器，用于统计该点的被观测次数，决定是否剔除坏点


class Scene:
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