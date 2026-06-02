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
    """Owns frames, 3D points, and registration state for one reconstruction."""

    def __init__(self, camera:Camera):
        self._camera = camera
        self._points = {}
        self._frames = {} 
        self._point_to_track = {}

        self._registered_ids = set()
        self._registered_sequence = []
        self._deferred_ids = set()
        self._failed_ids = set()
        self._failed_attempts = {}
        self._deferred_snapshots = {}
        self.max_register_attempts = 3

        self._frame_count = 0
        self._point_count = 0

    def get_intrisics(self):
        return self._camera.K

    def set_focal(self, focal):
        self._camera.update_focal_simple_pinhole(focal)

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
            logger.error(f"Frame directory not found: {e}")

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
        self._deferred_ids.discard(frame.idx)
        self._failed_attempts.pop(frame.idx, None)
        self._deferred_snapshots.pop(frame.idx, None)
        frame.is_registered = True

    def add_failed_frame(self, frame_idx):
        self._failed_ids.add(frame_idx)
        self._deferred_ids.discard(frame_idx)
        self._deferred_snapshots.pop(frame_idx, None)
        logger.warning(f"帧 {frame_idx} 被标记为失败状态")    

    def defer_frame(self, frame_idx):
        """Delay a failed frame so it can be retried after the map grows."""
        attempts = self._failed_attempts.get(frame_idx, 0) + 1
        self._failed_attempts[frame_idx] = attempts

        if attempts >= self.max_register_attempts:
            self.add_failed_frame(frame_idx)
            logger.warning(f"帧 {frame_idx} 已达到最大注册尝试次数 {attempts}，标记为永久失败。")
            return

        self._deferred_ids.add(frame_idx)
        self._deferred_snapshots[frame_idx] = (len(self._registered_ids), len(self._points))
        logger.info(f"帧 {frame_idx} 暂缓注册，当前尝试次数 {attempts}/{self.max_register_attempts}。")

    def is_deferred_retry_ready(self, frame_idx):
        """Return True when the map has grown since this frame was deferred."""
        if frame_idx not in self._deferred_ids:
            return False

        reg_count, point_count = self._deferred_snapshots.get(frame_idx, (0, 0))
        return len(self._registered_ids) > reg_count or len(self._points) > point_count

    @property
    def unregistered_frame_set(self):
        return self.candidate_frame_set - self._deferred_ids
    
    @property
    def deferred_frame_set(self):
        return self._deferred_ids - self._registered_ids - self._failed_ids
    
    @property
    def candidate_frame_set(self):
        return set([fid for fid in self._frames if (fid not in self._registered_ids) and (fid not in self._failed_ids)])
    
    @property
    def registered_frame_set(self):
        return self._registered_ids
    
    @property
    def failed_frame_set(self):
        return self._failed_ids
    
    def get_frame(self, frame_idx)->Frame:
        return self._frames.get(frame_idx)
    
    def get_registered_seq_list(self):
        return self._registered_sequence
    
    def all_frames(self):
        for fid in self._frames.keys():
            yield self._frames[fid]

    def create_point(self, track_idx, position3d, color=None):

        point_idx =  self._point_count
        point = Point(point_idx, track_idx, position3d, color)

        self._points[point.idx] = point
        self._point_count += 1
        self._point_to_track[point.idx] = track_idx

        return point.idx

    def create_points_from_info(self, point_info:list[tuple])->list[int]:
        """Create map points from tuples of (track_idx, position3d, color)."""
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
        
        track_idx = self._point_to_track.get(point_idx)
        
        if track_idx is not None:
            del self._point_to_track[point_idx]
        
        del self._points[point_idx]
        
        return track_idx
    

    def save_as_colmap(self, output_dir, track_manager):
        """Export cameras, images, and points in COLMAP text format."""
        from scipy.spatial.transform import Rotation as R_tool
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(output_path / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            K = self._camera.K
            f_val = K[0, 0]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"1 SIMPLE_PINHOLE {self._camera.width} {self._camera.height} {f_val} {cx} {cy}\n")

        # Precompute the observation-to-point lookup required by images.txt.
        feat_to_point = {}
        for pid, pt in self._points.items():
            track = track_manager.get_track_from_idx(pt.track_idx)
            for f_idx, feat_idx in track.observations:
                feat_to_point[(f_idx, feat_idx)] = pid

        with open(output_path / "images.txt", "w") as f:
            for f_idx in sorted(list(self._registered_ids)):
                frame = self._frames[f_idx]
                
                quat = R_tool.from_matrix(frame.R).as_quat()
                qw, qx, qy, qz = quat[3], quat[0], quat[1], quat[2]
                tx, ty, tz = frame.t.flatten()
                
                image_name = Path(frame.img_path).name
                f.write(f"{f_idx} {qw} {qx} {qy} {qz} {tx} {ty} {tz} 1 {image_name}\n")
                
                line_2 = []
                for i, kp in enumerate(frame.kps):
                    u, v = kp.pt
                    pid = feat_to_point.get((f_idx, i), -1)
                    line_2.append(f"{u} {v} {pid}")
                f.write(" ".join(line_2) + "\n")

        with open(output_path / "points3D.txt", "w") as f:
            for pid, pt in self._points.items():
                x, y, z = pt.position3d.flatten()
                r, g, b = np.clip(pt.color * 255, 0, 255).astype(int)
                error = 0.5
                
                track_data = []
                track = track_manager.get_track_from_idx(pt.track_idx)
                for f_idx, feat_idx in track.observations:
                    if f_idx in self._registered_ids:
                        track_data.append(f"{f_idx} {feat_idx}")
                
                f.write(f"{pid} {x} {y} {z} {r} {g} {b} {error} {' '.join(track_data)}\n")

        logger.info(f"成功将地图导出为 COLMAP 格式至: {output_dir}")
