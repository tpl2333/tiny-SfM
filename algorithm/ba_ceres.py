import numpy as np
import collections
import cv2
import logging
logger = logging.getLogger(__name__)

from management.worldmap import Map
from management.trackmanager import TrackManager

from build.Release import ba_core



class BundleAdjuster:
    """Python wrapper around the Ceres bundle-adjustment extension."""

    def __init__(self, worldmap:Map, trackmanager:TrackManager):
        self.map = worldmap
        self.tm = trackmanager

    def run_global_ba(self, fixed_frame_idx, is_fixed_focal = False):
        """Optimize all registered cameras and all current map points."""
        frame_ids = list(self.map._registered_ids)
        point_ids = list(self.map._points.keys())
        self._optimize(frame_ids, point_ids, fixed_frame_idx, is_fixed_focal, mode = "Global")

    def run_local_ba(self, window_size = 5):
        reg_seq = self.map._registered_sequence
        if len(reg_seq) < 3:
            return

        # Recent frames define the local active window.
        active_frame_ids = reg_seq[-window_size:]
        active_set = set(active_frame_ids)

        # Active points are points observed by the active frames. Older frames
        # that also observe them are anchor candidates.
        active_point_ids = set()
        anchor_counts = collections.Counter()

        for f_idx in active_frame_ids:
            feat_indices, pt_indices = self.tm.get_2d_3d_pairs(f_idx)
            for p_idx in pt_indices:
                active_point_ids.add(p_idx)
                
                track = self.tm._tracks.get(self.map._point_to_track[p_idx])
                if track:
                    for obs_f_idx, _ in track.observations:
                        if obs_f_idx not in active_set and obs_f_idx in self.map._registered_ids:
                            anchor_counts[obs_f_idx] += 1

        if not active_point_ids:
            return

        # Fix the strongest covisible old frames to keep the local window stable.
        most_common_anchors = anchor_counts.most_common(2)
        fixed_frame_ids = [f_idx for f_idx, count in most_common_anchors]
        
        if not fixed_frame_ids:
            fixed_frame_ids = [reg_seq[0]]

        all_involved_frames = list(active_set | set(fixed_frame_ids))
        point_ids = list(active_point_ids)

        self._optimize(
            frame_ids=all_involved_frames,
            point_ids=point_ids,
            fixed_frame_ids=fixed_frame_ids,
            is_fixed_focal=True,
            mode="Local"
        )


    def _optimize(self, frame_ids, point_ids, fixed_frame_ids, is_fixed_focal = True, mode="BA"):
        if not frame_ids or not point_ids:
            return

        # Ceres receives compact zero-based indices, not project ids.
        f_idx_to_ceres_idx = {f_idx: i for i, f_idx in enumerate(frame_ids)}
        fixed_frame_ceres_ids = [f_idx_to_ceres_idx[f_idx] for f_idx in fixed_frame_ids]

        # Camera parameter layout: angle-axis rotation followed by translation.
        cameras = []
        for f_idx in frame_ids:
            frame = self.map.get_frame(f_idx)
            rvec, _ = cv2.Rodrigues(frame.R)
            cam_param = np.hstack([rvec.flatten(), frame.t.flatten()])
            cameras.append(cam_param)
        cameras = np.array(cameras, dtype=np.float64)

        points = np.array([self.map.get_point(p_id).position3d for p_id in point_ids], dtype=np.float64)

        obs_data = []
        cam_ceres_indices = []
        pt_ceres_indices = []

        for p_ceres_idx, p_idx in enumerate(point_ids):
            track_idx = self.map.get_point(p_idx).track_idx
            track = self.tm._tracks[track_idx]
            for f_idx, feat_idx in track.observations:
                if f_idx in f_idx_to_ceres_idx:
                    obs_data.append(self.map.get_frame(f_idx).kps[feat_idx].pt)
                    cam_ceres_indices.append(f_idx_to_ceres_idx[f_idx])
                    pt_ceres_indices.append(p_ceres_idx)

        obs_data = np.array(obs_data, dtype=np.float64)
        cam_ceres_indices = np.array(cam_ceres_indices, dtype=np.int32)
        pt_ceres_indices = np.array(pt_ceres_indices, dtype=np.int32)

        fixed_frame_ceres_ids = np.array(fixed_frame_ceres_ids, dtype=np.int32)

        # The current backend optimizes one shared focal and fixed principal point.
        K = self.map.get_intrisics()
        cx, cy = K[0, 2], K[1, 2]
        shared_focal = np.array([K[0,0]], dtype=np.float64)

        logger.info(f"[{mode} BA] 优化中: {len(frame_ids)}个相机, {len(point_ids)}个点, {len(obs_data)}次观测...")

        # The pybind backend updates these contiguous arrays in place.
        cameras = np.ascontiguousarray(cameras)
        points = np.ascontiguousarray(points)
        shared_focal = np.ascontiguousarray(shared_focal)
        obs_data = np.ascontiguousarray(obs_data)
        cam_ceres_indices = np.ascontiguousarray(cam_ceres_indices)
        pt_ceres_indices = np.ascontiguousarray(pt_ceres_indices)
        fixed_frame_ceres_ids = np.ascontiguousarray(fixed_frame_ceres_ids)

        summary = ba_core.solve_ba_shared_focal(
            cameras, points, shared_focal,
            obs_data,
            cam_ceres_indices, pt_ceres_indices,
            fixed_frame_ceres_ids,
            is_fixed_focal,
            cx, cy
        )
        if "FAILURE" in summary:
            logger.error(f"[{mode} BA] {summary}")
        else:
            logger.info(f"[{mode} BA] {summary}")

        for i, f_idx in enumerate(frame_ids):
            opt_cam = cameras[i]
            R_new, _ = cv2.Rodrigues(opt_cam[:3])
            t_new = opt_cam[3:6].reshape(3, 1)
            self.map.get_frame(f_idx).set_pose(R_new, t_new)
        
        if not is_fixed_focal:
            self.map.set_focal(shared_focal[0])
        
        for i, p_id in enumerate(point_ids):      
            new_pos = points[i]
            if np.isnan(new_pos).any() or np.isinf(new_pos).any():
                continue 
            self.map.get_point(p_id).set_position3d(new_pos)

        logger.info(f"[{mode} BA] 优化完成！")
