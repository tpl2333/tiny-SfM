import cv2
import numpy as np
import scipy
import scipy.optimize 

from model.camera import Camera
from model.frame import Frame
from model.mappoint import Point
from model.worldmap import Map


class BA:

    def __init__(self, map:Map):

        self.map = map

        # 根据帧或点的id查询打包后位置
        # frame.idx/point.idx : params_position(int)
        self.frame_2_params_idx = {}
        self.point_2_params_idx = {}

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

        return np.array(pose_params+point_params,dtype=np.float64)

    def unpack_params(self, params):

        for frame_idx, start_idx in self.frame_2_params_idx.items():
            r_vec = params[start_idx : start_idx + 3]
            t_vec = params[start_idx + 3 : start_idx + 6]
            R, _ = cv2.Rodrigues(r_vec)
            self.map.frames[frame_idx].set_pose(R, t_vec.reshape(3, 1))

        for point_idx, start_idx in self.point_2_params_idx.items():
            self.map.points[point_idx].position = params[start_idx : start_idx + 3]

    def get_residuals(self, params):
        """计算所有观测的重投影误差"""
        residuals = []
        
        for point_idx, point in self.map.points.items():
            if point.is_bad: continue
            
            # 获取当前点的 3D 坐标
            p_idx = self.point_2_params_idx[point_idx]
            pw = params[p_idx : p_idx + 3]

            for frame_idx, feature_idx in point.observations.items():
                frame = self.map.frames[frame_idx]
                K = frame.camera.K
                
                # 获取该帧对应的 R 和 t
                if frame_idx == 0:
                    # 第一帧固定，直接用原始 R, t
                    R, t = frame.R, frame.t
                else:
                    f_idx = self.frame_2_params_idx[frame_idx]
                    r_vec = params[f_idx : f_idx + 3]
                    t = params[f_idx + 3 : f_idx + 6].reshape(3, 1)
                    R, _ = cv2.Rodrigues(r_vec)

                # 重投影过程：World -> Camera -> Image
                pc = R @ pw.reshape(3, 1) + t
                
                # 防止数值溢出或点在相机背面
                if pc[2, 0] < 1e-6:
                    residuals.extend([100.0, 100.0]) # 惩罚项
                    continue
                
                # 投影到像素坐标 (不考虑畸变)
                u_proj = K[0, 0] * (pc[0, 0] / pc[2, 0]) + K[0, 2]
                v_proj = K[1, 1] * (pc[1, 0] / pc[2, 0]) + K[1, 2]

                # 获取观测到的 2D 坐标
                u_obs, v_obs = frame.get_2d_position(feature_idx)
                
                residuals.append(u_proj - u_obs)
                residuals.append(v_proj - v_obs)

        return np.array(residuals)

    def optimize(self):
        """执行优化"""
        x0 = self.pack_params()
        
        # 使用更具鲁棒性的损失函数 (loss='soft_l1') 来抑制外点
        res = scipy.optimize.least_squares(
            self.get_residuals, 
            x0, 
            method='trf', 
            loss='soft_l1', 
            f_scale=1.0, # 这里的 f_scale 相当于像素误差的阈值
            verbose=2
        )
        
        self.unpack_params(res.x)
        print(f"BA 优化完成。初始误差: {np.linalg.norm(res.fun):.2f}")                

                

    



