import cv2
import numpy as np
from model.camera import Camera

class Point:
    """Triangulated 3D point owned by a feature track."""

    def __init__(self, point_idx, track_idx, position3d, color=None):

        self.idx = point_idx
        self.track_idx = track_idx

        self._position3d = position3d
        self._color = self._normalize_color(color)

        self.is_bad = False
        self.optimize_count = 0
    
    @property
    def position3d(self):
        return self._position3d
    
    @property
    def color(self):
        return self._color

    def _normalize_color(self, color):
        """Store point color as RGB float values in [0, 1]."""
        if color is None:
            return np.array([0.5, 0.5, 0.5], dtype=np.float64)

        color = np.asarray(color, dtype=np.float64).reshape(3)
        if np.max(color) > 1.0:
            color = color / 255.0
        return np.clip(color, 0.0, 1.0)
    
    def set_position3d(self, pts3d, by_optimization = False):
        """Update point coordinates, optionally counting BA updates."""
        self._position3d = pts3d
        if by_optimization:
            self.optimize_count += 1
