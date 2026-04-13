import cv2
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)


from model.camera import Camera
from model.mappoint import Point
from model.frame import Frame
from management.viewgraph import EdgeData

class Map:

    def __init__(self, camera:Camera):
        # -----核心数据-----
        self._camera = camera
        # self.points = {point.idx: Point}
        self._points = {}
        # self.frames = {frame.idx: Frame}
        self._frames = {} 
        # self._point_to_track = {point.idx: track.idx}
        self._point_to_track = {}

        # registered_idx = set(registered_frame.idx)
        self._registered_ids = set()
        # 
        self._failed_ids = set()

        # -----帧与点的id初始化-----
        self._frame_count = 0
        self._point_count = 0

    # -------相机相关方法-------
    def get_intrisics(self):
        return self._camera.K

    # -------帧相关方法-------
    def add_frame(self, img_path):

        frame_idx = self._frame_count
        frame = Frame(img_path, frame_idx, self._camera)
        self._frames[frame.idx] = frame
        self._frame_count += 1
            
    def load_frame_dir(self, img_dir):
        allowed_suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        dir_path = Path(img_dir)
        try:
            for item in dir_path.iterdir():
                if item.is_file() and item.suffix.lower() in allowed_suffixes:
                    self.add_frame(str(item))
        except FileNotFoundError as e:
            print(f"[Map] Frame Directory Not Found!:{e}")

        logger.info(f"共有 {len(self._frames)} 张图像加载成功")

    def register_frame(self, frame_idx, R = np.eye(3), t = np.zeros((3, 1))):

        if frame_idx not in self._frames:
            logger.error(f"[Map] Frame ID {frame_idx} not in Map")
            raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")
        if frame_idx in self._registered_ids:
            logger.warning(f"Frame {frame_idx} has already registered, update the pose")

        frame = self._frames[frame_idx]
        frame.set_pose(R, t)
        self._registered_ids.add(frame.idx)
        frame.is_registered = True

    def add_failed_frame(self, frame_idx):
        self._failed_ids.add(frame_idx)
        logger.warning(f"帧 {frame_idx} 被标记为失败状态")    

    @property
    def unregistered_frame_set(self):
        # unregistered_frames = set(List[unregistered_frame.idx])
        return set([fid for fid in self._frames if (fid not in self._registered_ids) and (fid not in self._failed_ids)])
    
    @property
    def registered_frame_set(self):
        # registered_frames = set(registered_frame.idx)
        return self._registered_ids
    
    @property
    def failed_frame_set(self):
        # failed_frames = set(failed_frame.idx)
        return self._failed_ids
    
    def get_frame(self, frame_idx)->Frame:
        return self._frames.get(frame_idx)
    
    def all_frames(self):
        for fid in self._frames.keys():
            yield self._frames[fid]

    # -------地图点相关方法------- 
    def create_point(self, track_idx, position3d, color=None):

        point_idx =  self._point_count
        point = Point(point_idx, track_idx, position3d, color)

        self._points[point.idx] = point
        self._point_count += 1

        return point.idx

    def create_points_from_info(self, point_info:list[tuple])->list[int]:
        """ 从三角化得到的点信息创建地图点对象，返回点索引列表

        Args:
            point_info (list[tuple]): [(track_idx, position3d, color)]

        Returns:
            point_indice (list[int]): [point1.idx,.....]
        """
        point_indice = []
        for track_idx, x, color in point_info:
            point_idx = self.create_point(track_idx, x, color)
            point_indice.append(point_idx)
        
        return point_indice
    
    def get_point(self, point_idx)->Point:
        return self._points.get(point_idx)


    
    # def add_observation(self, point_idx, frame_idx, feature_idx):

    #     if point_idx not in self._points:
    #         raise KeyError(f"[Map] Point ID {point_idx} not in Map")
    #     if frame_idx not in self._frames:
    #         raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")

    #     point = self._points[point_idx]
    #     frame = self._frames[frame_idx]

    #     point.add_observation(frame_idx, feature_idx)
    #     frame.add_observation(feature_idx, point_idx)
        
