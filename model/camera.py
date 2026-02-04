import numpy as np
import cv2
from enum import Enum

class CameraSource(Enum):
    """
    标记相机内参的来源信度
    """
    UNKNOWN = 0     # 未初始化
    GUESS = 1       # 猜测值 (信度低，BA优化时通常需要调整)
    CALIBRATED = 2  # 标定值 (信度高，BA优化时通常固定或仅微调)
    OPTIMIZED = 3   # 优化值 (来自BA的结果，信度最高)

class Camera:
    def __init__(self, width=None, height=None):
        """
        初始化相机对象，必须知道图像尺寸。
        """
        self.width = width
        self.height = height
        
        # 核心数据：内参矩阵 K (3x3) 和 畸变系数 D (1x5 or 1xN)
        self._K = np.eye(3, dtype=np.float64)
        self._dist = np.zeros((5, 1), dtype=np.float64) # 默认无畸变
        
        # 状态标志位
        self.source = CameraSource.UNKNOWN
        
        # 锁定机制：如果为 True，在优化阶段不应该修改此相机的参数
        self.locked = False 

    def set_size(self, height, width):
        """
        设置图像尺寸，防止和frame冲突
        """
        if self.width is not None and (self.width != width or self.height != height):
            print(f"[Camera] Warning: Image size changed from {self.width}x{self.height} to {width}x{height}!")
            # 在某些变焦或裁切场景下可能合法，但通常意味着错误
        
        self.width = width
        self.height = height

    # 初始化途径一：猜
    def setup_by_guess(self, fov_scale=0.8):
        """
        接口1：当没有标定数据时，根据图像尺寸粗略估计内参。
        通常假设光心在图像中心，焦距为图像宽度的 1.2 倍左右。
        """
        if self.width is None or self.height is None:
            raise ValueError("[Camera] Cannot guess intrinsics without image size! Call set_size() first.")
        
        focal_length = self.width * fov_scale
        cx = self.width / 2.0
        cy = self.height / 2.0
        
        self._K = np.array([
            [focal_length, 0, cx],
            [0, focal_length, cy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        self._dist = np.zeros((5, 1), dtype=np.float64) # 猜测时通常假设无畸变
        
        self.source = CameraSource.GUESS
        self.locked = False 
        print(f"[Camera] Initialized by GUESS. K:\n{self._K}")

    # 初始化途径二：标定
    def setup_by_calibration(self, height, width, K, dist, lock_it=True):
        """
        接口2：接收来自 cv2.calibrateCamera 的结果。
        """
        self.set_size(height, width)

        if self.width is None or self.height is None:
            raise ValueError("[Camera] image size of calibrated intrinsics is None! Call set_size() first.")

        self._K = np.array(K, dtype=np.float64)
        self._dist = np.array(dist, dtype=np.float64)
        
        self.source = CameraSource.CALIBRATED

        self.locked = lock_it 
        print(f"[Camera] Initialized by CALIBRATION. Locked={self.locked}")

    # 优化途径：来自整体优化 (留给 BA 的接口)
    def get_params_vector(self):
        """
        [导出接口]
        将内参“扁平化”为一个向量，供优化器（如 scipy.optimize 或 Ceres）使用。
        通常优化器只认 vector，不认 matrix。
        返回格式示例: [fx, fy, cx, cy, k1, k2, p1, p2]
        """
        # 提取 fx, fy, cx, cy
        fx = self._K[0, 0]
        fy = self._K[1, 1]
        cx = self._K[0, 2]
        cy = self._K[1, 2]
        
        # 提取畸变 (假设是标准的5参数模型)
        d = self._dist.flatten()
        
        # 拼接
        params = np.array([fx, fy, cx, cy, d[0], d[1], d[2], d[3], d[4]])
        return params

    def update_from_optimization(self, params_vector):
        """
        [导入接口]
        接收优化器计算出的新向量，更新内部状态。
        """
        if self.locked:
            print("[Camera] Warning: Attempting to update a LOCKED camera. Ignored.")
            return

        fx, fy, cx, cy = params_vector[0:4]
        self._K[0, 0] = fx
        self._K[1, 1] = fy
        self._K[0, 2] = cx
        self._K[1, 2] = cy
        
        self._dist = params_vector[4:].reshape(-1, 1)
        
        self.source = CameraSource.OPTIMIZED

    @property
    def K(self):
        """只读属性，防止外部直接修改 self.K = ... 破坏状态"""
        return self._K

    @property
    def dist(self):
        return self._dist
