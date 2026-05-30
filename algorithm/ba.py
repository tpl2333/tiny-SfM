import cv2
import numpy as np
import scipy
import scipy.optimize 
import logging
from scipy.sparse import lil_matrix

from model.camera import Camera
from model.frame import Frame
from model.mappoint import Point
from management.worldmap import Map

logger = logging.getLogger(__name__)


class BA:
    """Legacy SciPy bundle-adjustment implementation.

    The active pipeline uses algorithm.ba_ceres.BundleAdjuster. This class is
    kept as a readable Python reference for residual construction and sparsity.
    """

    def __init__(self, map:Map):

        self.map = map

        # Project ids are mapped to offsets inside the flattened parameter vector.
        self.frame_2_params_idx = {}
        self.point_2_params_idx = {}
        self.intrisic_2_params_idx = None

    def pack_params(self):

        pose_params = []
        for frame_idx, frame in self.map.frames.items():
            if frame_idx == 0:
                continue

            self.frame_2_params_idx[frame_idx] = len(pose_params)
            r_vec, _ = cv2.Rodrigues(frame.R)
            pose_vec = np.concatenate([r_vec.reshape(3),frame.t.reshape(3)])
            pose_params.extend(pose_vec)

        point_params = []
        for point_idx, point in self.map.points.items():
            if point.is_bad:
                continue

            self.point_2_params_idx[point_idx] = len(pose_params)+len(point_params)
            point_params.extend(point.position)

        shared_camera = list(self.map.frames.values())[0].camera
        if not shared_camera.is_locked:
            intrisic_vec = list(shared_camera.get_params_vector())
            self.intrisic_2_params_idx = len(pose_params)+len(point_params)
            return np.array(pose_params+point_params+intrisic_vec,dtype=np.float64)
        else:
            return np.array(pose_params+point_params,dtype=np.float64)

    def unpack_params(self, params):

        for frame_idx, start_idx in self.frame_2_params_idx.items():
            r_vec = params[start_idx : start_idx + 3]
            t_vec = params[start_idx + 3 : start_idx + 6]
            R, _ = cv2.Rodrigues(r_vec)
            self.map.frames[frame_idx].set_pose(R, t_vec.reshape(3, 1))

        for point_idx, start_idx in self.point_2_params_idx.items():
            self.map.points[point_idx].position = params[start_idx : start_idx + 3]
        
        if self.intrisic_2_params_idx is not None:
            cam_vec = params[self.intrisic_2_params_idx : self.intrisic_2_params_idx + 9]
            list(self.map.frames.values())[0].camera.update_from_optimization(cam_vec)

    def get_residuals(self, params):
        """Compute all reprojection residuals for least-squares optimization."""
        residuals = []
        if self.intrisic_2_params_idx is not None:
            intrisic = params[self.intrisic_2_params_idx:]
            fx, fy, cx, cy = intrisic[0], intrisic[1], intrisic[2], intrisic[3]
            if list(self.map.frames.values())[0].camera.is_dist:
                k1, k2, p1, p2, k3 = intrisic[4], intrisic[5], intrisic[6], intrisic[7], intrisic[8]
        
        for point_idx, point in self.map.points.items():
            if point.is_bad: continue
            
            p_idx = self.point_2_params_idx[point_idx]
            pw = params[p_idx : p_idx + 3]

            for frame_idx, feature_idx in point.observations.items():
                frame = self.map.frames[frame_idx]

                if frame_idx == 0:
                    R, t = frame.R, frame.t
                else:
                    f_idx = self.frame_2_params_idx[frame_idx]
                    r_vec = params[f_idx : f_idx + 3]
                    t = params[f_idx + 3 : f_idx + 6].reshape(3, 1)
                    R, _ = cv2.Rodrigues(r_vec)

                pc = R @ pw.reshape(3, 1) + t
                
                if pc[2, 0] < 1e-6:
                    residuals.extend([100.0, 100.0]) 
                    continue
                
                z = pc[2, 0]
                x_n, y_n = pc[0,0]/z, pc[1,0]/z 

                if self.intrisic_2_params_idx is None:
                    K = frame.camera.K
                    u_proj = K[0, 0] * x_n + K[0, 2]
                    v_proj = K[1, 1] * y_n + K[1, 2]
                elif list(self.map.frames.values())[0].camera.is_dist:
                    r2 = x_n**2 + y_n**2
                    r4 = r2**2
                    r6 = r2**3
                    radial = (1 + k1*r2 + k2*r4 + k3*r6) 
                    x_dist = x_n * radial + (2*p1*x_n*y_n + p2*(r2 + 2*x_n**2))
                    y_dist = y_n * radial + (p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n)

                    u_proj = fx * x_dist + cx
                    v_proj = fy * y_dist + cy
                else:
                    u_proj = fx * x_n + cx
                    v_proj = fy * y_n + cy

                u_obs, v_obs = frame.get_2d_position(feature_idx)
                
                residuals.append(u_proj - u_obs)
                residuals.append(v_proj - v_obs)

        return np.array(residuals)

    def optimize(self):
        """Run sparse nonlinear least squares and write results back to the map."""
        x0 = self.pack_params()

        sparsity = self.get_sparsity_matrix(x0)
        
        res = scipy.optimize.least_squares(
            self.get_residuals, 
            x0, 
            jac_sparsity=sparsity,
            x_scale='jac',
            method='trf', 
            loss='soft_l1', 
            f_scale=1.0,
            verbose=0
        )
        
        self.unpack_params(res.x)
        logger.info(f"BA 优化完成。残差范数: {np.linalg.norm(res.fun):.2f}") 

    def get_sparsity_matrix(self, params):
        n_res = 0
        for pt in self.map.points.values():
            if pt.is_bad: continue
            n_res += len(pt.observations) * 2
        
        n_params = len(params)
        sparsity = lil_matrix((n_res, n_params), dtype=int)
        
        res_idx = 0
        for pt_idx, pt in self.map.points.items():
            if pt.is_bad: continue
            p_start = self.point_2_params_idx[pt_idx]
            
            for frame_idx, _ in pt.observations.items():
                sparsity[res_idx : res_idx + 2, p_start : p_start + 3] = 1
                
                if frame_idx in self.frame_2_params_idx:
                    f_start = self.frame_2_params_idx[frame_idx]
                    sparsity[res_idx : res_idx + 2, f_start : f_start + 6] = 1
                
                if self.intrisic_2_params_idx is not None:
                    sparsity[res_idx : res_idx + 2, self.intrisic_2_params_idx : ] = 1
                    
                res_idx += 2
                
        return sparsity 
    
    def calculate_rmse(self, params=None):
        """Report reprojection error statistics in pixels."""
        if params is None:
            residuals = self.get_residuals(self.pack_params())
        else:
            residuals = self.get_residuals(params)

        num_observations = len(residuals) / 2
        
        mse = np.mean(np.square(residuals))
        rmse = np.sqrt(mse)
        
        res_reshaped = residuals.reshape(-1, 2)
        l2_norms = np.linalg.norm(res_reshaped, axis=1)
        mean_error = np.mean(l2_norms)
        max_error = np.max(l2_norms)

        logger.info(
            "BA 质量评估报告:\n"
            f"观测点总数: {int(num_observations)}\n"
            f"RMSE (像素): {rmse:.4f}\n"
            f"平均误差 (像素): {mean_error:.4f}\n"
            f"最大偏差 (像素): {max_error:.4f}"
        )
        
        return rmse             

                

    



