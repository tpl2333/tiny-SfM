from __future__ import annotations
import cv2
import numpy as np
from model.camera import Camera
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

        self.is_bad = False
    
    def add_observation(self, frame:Frame, feature_idx):

        if frame.idx not in self.observations:
            self.observations[frame.idx] = feature_idx
            # 可以在这里增加计数器，用于统计该点的被观测次数，决定是否剔除坏点

