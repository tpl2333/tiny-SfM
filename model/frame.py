import numpy as np
import cv2
from model.camera import Camera

class Frame:
    """Image frame with features and a world-to-camera pose."""

    def __init__(self, img_path, frame_idx, camera:Camera):

        self.idx = frame_idx
        self.img_path = img_path

        self._img = cv2.imread(img_path)
        self._camera = camera  
        if self._img is None:
            raise ValueError(f"[Frame] Image path error! Cannot read {self.img_path}")
        
        self._kps = None
        self._des = None
        self._colors = None

        # Pose convention: X_camera = R * X_world + t.
        self._R = np.eye(3) 
        self._t = np.zeros((3, 1))

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
    
    @property
    def R(self):
        return self._R
    
    @property
    def t(self):
        return self._t
    
    @property
    def height(self):
        return self._img.shape[0]
    
    @property
    def weight(self):
        return self._img.shape[1]


    def set_feature(self, kps, des):
        self._kps = kps
        self._des = des

    def set_pose(self, R, t):
        self._R = R
        self._t = t

    def get_proj_matrix(self):
        """Return the 3x4 projection matrix P = K [R|t]."""
        t_vec = self._t.reshape(3, 1)
        Rt = np.hstack((self._R, t_vec))
        P = np.dot(self._camera.K, Rt)
        return P

    def get_center(self):
        """Return the camera center in world coordinates."""
        return -np.dot(self._R.T, self._t)

    def get_2d_position(self, feature_idx):
        """Return the observed pixel position of a keypoint as (u, v)."""
        return np.array(self._kps[feature_idx].pt, dtype=np.float64) 
    
    def get_color(self, u, v):
        """Return the BGR image color at pixel coordinates (u, v)."""
        col = int(np.clip(u, 0, self.weight - 1))
        row = int(np.clip(v, 0, self.height - 1))
        return self._img[row, col]
