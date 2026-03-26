import numpy as np
import cv2
from model.camera import Camera

class Frame:
    # Frame idx
    _counter = 0

    def __init__(self, img_path, camera:Camera):

        self.idx = Frame._counter
        Frame._counter += 1

        self._img = cv2.imread(img_path)
        self._camera = camera  
        if self._img is None:
            raise ValueError(f"[Frame] Image path error! Cannot read {self.img_path} ")
        
        self._kps = None  # 关键点 (N, 2)
        self._des = None  # 描述子 (N, D)
        self._colors = None # 颜色 用于点云渲染

        # T_cw: World -> Camera
        # T_cw = [R|t]
        self._R = np.eye(3) 
        self._t = np.zeros((3, 1))

        # feature_2_point = dict{feature.idx：Point.idx} 
        # feature_idx: 当前帧特征点(Keypoint)的局部索引。
        # point_idx: 关联的三维点(MapPoint)的全局唯一 ID。
        self.feature_2_point = {}

        # 是否注册到地图中
        self.is_registered = False

    @property
    def camera(self):
        return self._camera
    
    @property
    def kps(self):
        if self._kps is None:
            return [] 
        return self._kps
    
    @property
    def des(self):
        if self._des is None:
            return []
        return self._des


    def set_feature(self, kps, des):
        self._kps = kps
        self._des = des

    def set_pose(self, R, t):
        self._R = R
        self._t = t

    def get_proj_matrix(self):
        """
        获取 3x4 投影矩阵 P = K[R|t]
        """
        # 确保 t 是 (3,1)
        t_vec = self._t.reshape(3, 1)
        # 拼接 [R|t]
        Rt = np.hstack((self._R, t_vec))
        # 乘内参 K
        P = np.dot(self._camera.K, Rt)
        return P

    def get_center(self):
        """
        获取相机在世界坐标系下的中心坐标 (用于可视化相机轨迹)
        公式: C = -R^T * t
        """
        return -np.dot(self._R.T, self._t)

    def get_2d_position(self, feature_idx):
        """
        获取指定特征点在图像平面上的 2D 观测坐标 (u, v)。
        """
        return np.array(self._kps[feature_idx].pt, dtype=np.float64) 
    
    def add_observation(self, feature_idx, point_idx):
        """
        建立 2D 特征点与全局 3D 地图点的关联映射。

        即更新self.feature_2_point = {feature_idx:point_idx}
            feature_idx: 当前帧特征点(Keypoint)的局部索引。
            point_idx: 关联的三维点(MapPoint)的全局唯一 ID。
        """
    
        if feature_idx in self.feature_2_point:
            if self.feature_2_point[feature_idx] != point_idx:
                raise ValueError(f"[Frame] 帧{self.idx} 的特征点 {feature_idx} 已经关联了点 {self.feature_2_point[feature_idx]}，不能重复关联点 {point_idx}")
            return
        
        self.feature_2_point[feature_idx] = point_idx

    def get_observed_point(self, feature_idx):
        """ 查询帧内特征索引所对应的全局 3D 地图点

        Args:
            feature_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        if feature_idx in self.feature_2_point:
            return self.feature_2_point[feature_idx]
        else:
            return None

    def get_color(self, u, v):

        return self._img[u,v]

