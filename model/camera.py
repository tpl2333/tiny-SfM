import numpy as np
import cv2
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class CameraSource(Enum):
    """Origin of the intrinsic parameters."""
    UNKNOWN = 0
    GUESS = 1
    CALIBRATED = 2
    OPTIMIZED = 3

class Camera:
    def __init__(self, width=None, height=None, is_dist = False):
        """Create a shared pinhole camera model for all frames."""
        self.width = width
        self.height = height
        self.is_dist = is_dist
        
        # K is the active intrinsic matrix; dist is reserved for calibrated input.
        self._K = np.eye(3, dtype=np.float64)
        self._dist = np.zeros((5, 1), dtype=np.float64)
        
        self.source = CameraSource.UNKNOWN
        
        # Locked intrinsics are not updated by BA.
        self.is_locked = False 

    def set_size(self, height, width):
        """Set image dimensions and warn if they change after initialization."""
        if self.width is not None and (self.width != width or self.height != height):
            logger.warning(f"Image size changed from {self.width}x{self.height} to {width}x{height}")
        
        self.width = width
        self.height = height

    def setup_by_guess(self, fov_scale=1.2, lock_it = False):
        """Initialize intrinsics from image size when calibration is unavailable."""
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
        
        self._dist = np.zeros((5, 1), dtype=np.float64)
        
        self.source = CameraSource.GUESS
        
        if lock_it:
            self.is_locked = True 
            logger.info(f"Initialized camera by guess. Locked={self.is_locked}")
        else:
            self.is_locked = False 
            logger.info(f"Initialized camera by guess. K:\n{self._K}")

    def setup_by_calibration(self, height, width, K, dist, lock_it=True):
        """Initialize intrinsics from an external calibration result."""
        self.set_size(height, width)

        if self.width is None or self.height is None:
            raise ValueError("[Camera] image size of calibrated intrinsics is None! Call set_size() first.")

        self._K = np.array(K, dtype=np.float64)
        self._dist = np.array(dist, dtype=np.float64)
        
        self.source = CameraSource.CALIBRATED
        
        if lock_it:
            self.is_locked = True 
            logger.info(f"Initialized camera by calibration. Locked={self.is_locked}")

    def update_focal_simple_pinhole(self, focal):
        """Update the shared focal length after SIMPLE_PINHOLE BA."""
        if self.is_locked:
            logger.warning("Attempted to update a locked camera. Ignored.")
            return

        self._K[0, 0] = focal
        self._K[1, 1] = focal
   
        self.source = CameraSource.OPTIMIZED

    @property
    def K(self):
        """Return the active 3x3 intrinsic matrix."""
        return self._K

    @property
    def dist(self):
        return self._dist
