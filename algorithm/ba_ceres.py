import cv2
import numpy as np

from model.worldmap import Map

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "build/Release"))
try:
    import ba_core
except ImportError as e:
    print(f"[ba_ceres]: ba_core导入失败: {e}")

class Optimizer:

    def __init__(self, worldmap:Map):

        self.worldmap = worldmap

    def load