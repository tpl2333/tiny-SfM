import numpy as np
import cv2
from camera import Camera

class Frame:
    # Frame idx
    _counter = 0

    def __init__(self, img_path, camera:Camera):
        """
        idx: 帧的全局唯一索引 (int)
        img_path: 图像路径
        camera: 关联的 Camera 对象 (引用)
        """

        self.idx = Frame._counter
        Frame._counter += 1

        self.img_path = img_path
        self.img = cv2.imread(self.img_path)
        self.camera = camera  

        if self.img is None:
            raise ValueError(f"[Frame] Image path error! Cannot read {self.img_path} ")
        
        # 特征数据 
        # 这一步可以在初始化时提取，也可以懒加载
        self.kps = None  # 关键点 (N, 2)
        self.des = None  # 描述子 (N, D)
        
        # 颜色信息，用于后续点云着色
        self.colors = None 

        # 外参——世界坐标系到相机坐标系
        # T_cw: World -> Camera
        # P = K[R|t]
        self.R = np.eye(3) 
        self.t = np.zeros((3, 1))

        # 此图像注册的3D点
        # index：MapPoint (index来自Frame.kps[index])
        self.map_point = {}

        # 标记该帧是否已经注册到地图中
        self.is_registered = False 

    def set_pose(self, R, t):
        """
        设置外参
        """
        self.R = R
        self.t = t
        self.is_registered = True

    def get_proj_matrix(self):
        """
        获取 3x4 投影矩阵 P = K[R|t]
        """
        # 确保 t 是 (3,1)
        t_vec = self.t.reshape(3, 1)
        # 拼接 [R|t]
        Rt = np.hstack((self.R, t_vec))
        # 乘内参 K
        P = np.dot(self.camera.K, Rt)
        return P

    def get_center(self):
        """
        获取相机在世界坐标系下的中心坐标 (用于可视化相机轨迹)
        公式: C = -R^T * t
        """
        return -np.dot(self.R.T, self.t)