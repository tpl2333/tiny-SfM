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
        self._registered_sequence = []
        # 
        self._failed_ids = set()

        # -----帧与点的id初始化-----
        self._frame_count = 0
        self._point_count = 0

    # -------相机相关方法-------
    def get_intrisics(self):
        return self._camera.K

    def set_focal(self, focal):
        self._camera.update_focal_simple_pinhole(focal)

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
        self._registered_sequence.append(frame.idx)
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
    
    def get_registered_seq_list(self):
        return self._registered_sequence
    
    def all_frames(self):
        for fid in self._frames.keys():
            yield self._frames[fid]

    # -------地图点相关方法------- 
    def create_point(self, track_idx, position3d, color=None):

        point_idx =  self._point_count
        point = Point(point_idx, track_idx, position3d, color)

        self._points[point.idx] = point
        self._point_count += 1
        self._point_to_track[point.idx] = track_idx

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
    
    def remove_point(self, point_idx):

        if point_idx not in self._points:
            return None
        
        # 1. 获取对应的轨迹 ID
        track_idx = self._point_to_track.get(point_idx)
        
        # 2. 从各个容器中清理
        if track_idx is not None:
            del self._point_to_track[point_idx]
        
        del self._points[point_idx]
        
        return track_idx
    

    def save_as_colmap(self, output_dir, track_manager):
        """
        将当前的重建结果保存为 COLMAP 文本格式 (.txt)
        """
        from scipy.spatial.transform import Rotation as R_tool
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. 保存 cameras.txt
        # 格式: CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
        with open(output_path / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            K = self._camera.K
            f_val = K[0, 0]
            cx, cy = K[0, 2], K[1, 2]
            # 这里假设是简单的 PINHOLE 模型 (fx, fy, cx, cy)
            f.write(f"1 SIMPLE_PINHOLE {self._camera.width} {self._camera.height} {f_val} {cx} {cy}\n")

        # 2. 预处理：构建 (frame_idx, feat_idx) -> point3D_id 的映射
        # 这是为了在 images.txt 中快速填入特征点关联的 3D 点 ID
        feat_to_point = {}
        for pid, pt in self._points.items():
            track = track_manager.get_track_from_idx(pt.track_idx)
            for f_idx, feat_idx in track.observations:
                feat_to_point[(f_idx, feat_idx)] = pid

        # 3. 保存 images.txt
        # 格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        #      POINTS2D[] (x, y, point3D_id)
        with open(output_path / "images.txt", "w") as f:
            for f_idx in sorted(list(self._registered_ids)):
                frame = self._frames[f_idx]
                
                # R 转换为四元数 [w, x, y, z]
                quat = R_tool.from_matrix(frame.R).as_quat() # 返回 [x, y, z, w]
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                tx, ty, tz = frame.t.flatten()
                
                image_name = Path(frame.img_path).name
                f.write(f"{f_idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n")
                
                # 写入该帧所有的特征点及其对应的 3D 点 ID
                line_2 = []
                for i, kp in enumerate(frame.kps):
                    u, v = kp.pt
                    # 如果该特征点没有关联 3D 点，填 -1
                    pid = feat_to_point.get((f_idx, i), -1)
                    line_2.append(f"{u} {v} {pid}")
                f.write(" ".join(line_2) + "\n")

        # 4. 保存 points3D.txt
        # 格式: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] (IMAGE_ID, FEAT_IDX)
        with open(output_path / "points3D.txt", "w") as f:
            for pid, pt in self._points.items():
                x, y, z = pt.position3d.flatten()
                # 颜色转回 0-255 整数
                r, g, b = (pt.color * 255).astype(int)
                error = 0.5 # 占位符，如果之前没存平均误差的话
                
                # 寻找观测到该点的所有已注册帧
                track_data = []
                track = track_manager.get_track_from_idx(pt.track_idx)
                for f_idx, feat_idx in track.observations:
                    if f_idx in self._registered_ids:
                        track_data.append(f"{f_idx} {feat_idx}")
                
                f.write(f"{pid} {x} {y} {z} {r} {g} {b} {error} {' '.join(track_data)}\n")

        logger.info(f"成功将地图导出为 COLMAP 格式至: {output_dir}")


    
    # def add_observation(self, point_idx, frame_idx, feature_idx):

    #     if point_idx not in self._points:
    #         raise KeyError(f"[Map] Point ID {point_idx} not in Map")
    #     if frame_idx not in self._frames:
    #         raise KeyError(f"[Map] Frame ID {frame_idx} not in Map")

    #     point = self._points[point_idx]
    #     frame = self._frames[frame_idx]

    #     point.add_observation(frame_idx, feature_idx)
    #     frame.add_observation(feature_idx, point_idx)
        
