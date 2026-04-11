import cv2
import numpy as np
from model.camera import Camera

class Point:

    def __init__(self, point_idx, track_idx, position3d, color=None):

        self.idx = point_idx
        self.track_idx = track_idx

        self._position3d = position3d
        self._color = color
        
        # # observations = dict{frame.idx: feature_idx}
        # # frame_idx: 关联的帧(Frame)的全局唯一 ID。
        # # feature_idx: 当前帧特征点(Keypoint)的局部索引。
        # self.observations = {}

        self.is_bad = False
    
    @property
    def position3d(self):
        return self._position3d
    
    @property
    def color(self):
        return self._color
    
    # def add_observation(self, frame_idx, feature_idx):
    #     """
    #     建立该 3D 地图点 与 图像帧与帧中的特征 2D 像素点的关联映射。

    #     即更新self.observation = {frame_idx:feature_idx}
    #         frame_idx: 关联的帧(Frame)的全局唯一 ID。
    #         feature_idx: 当前帧特征点(Keypoint)的局部索引。
    #     """

    #     if frame_idx in self.observations:
    #         if self.observations[frame_idx] != feature_idx:
    #             raise ValueError(f"[Point] 当前地图点 {self.idx} 已经被帧 {frame_idx} 中的关键点 {self.observations[frame_idx]} 所关联，不能重复关联该帧下的关键点 {feature_idx}")

    #     self.observations[frame_idx] = feature_idx
    #     # 可以在这里增加计数器，用于统计该点的被观测次数，决定是否剔除坏点

