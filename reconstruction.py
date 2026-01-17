import cv2
import numpy as np
from matching import FeatureMatcher
from camera import Camera
from frame import Frame

class Reconstruction:
    def __init__(self, frame1:Frame, frame2:Frame, matches:list, model_type:str, model_matrix:np):
        asasas