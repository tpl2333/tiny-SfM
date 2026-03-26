import cv2
import numpy as np
import scipy
import scipy.optimize 
from scipy.sparse import lil_matrix

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
        self.intrisic_2_params_idx = None # 内参起始点

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
        """计算所有观测的重投影误差"""
        residuals = []
        if self.intrisic_2_params_idx is not None:
            intrisic = params[self.intrisic_2_params_idx:]
            fx, fy, cx, cy = intrisic[0], intrisic[1], intrisic[2], intrisic[3]
            if list(self.map.frames.values())[0].camera.is_dist:
                k1, k2, p1, p2, k3 = intrisic[4], intrisic[5], intrisic[6], intrisic[7], intrisic[8]
        
        for point_idx, point in self.map.points.items():
            if point.is_bad: continue
            
            # 获取当前点的 3D 坐标
            p_idx = self.point_2_params_idx[point_idx]
            pw = params[p_idx : p_idx + 3]

            for frame_idx, feature_idx in point.observations.items():
                frame = self.map.frames[frame_idx]

                # 获取该帧对应的 R 和 t
                if frame_idx == 0:
                    # 第一帧固定，直接用原始 R, t
                    R, t = frame.R, frame.t
                else:
                    f_idx = self.frame_2_params_idx[frame_idx]
                    r_vec = params[f_idx : f_idx + 3]
                    t = params[f_idx + 3 : f_idx + 6].reshape(3, 1)
                    R, _ = cv2.Rodrigues(r_vec)

                # 到相机坐标系：World -> Camera
                pc = R @ pw.reshape(3, 1) + t
                
                # 防止数值溢出或点在相机背面，加入惩罚项
                if pc[2, 0] < 1e-6:
                    residuals.extend([100.0, 100.0]) 
                    continue
                
                # 归一化
                z = pc[2, 0]
                x_n, y_n = pc[0,0]/z, pc[1,0]/z 

                # 对于frame共用内参的优化
                if self.intrisic_2_params_idx is None:
                    #无畸变，不优化
                    K = frame.camera.K
                    u_proj = K[0, 0] * x_n + K[0, 2]
                    v_proj = K[1, 1] * y_n + K[1, 2]
                elif list(self.map.frames.values())[0].camera.is_dist:
                    # 有畸变，需要优化
                    r2 = x_n**2 + y_n**2
                    r4 = r2**2
                    r6 = r2**3
                    radial = (1 + k1*r2 + k2*r4 + k3*r6) 
                    x_dist = x_n * radial + (2*p1*x_n*y_n + p2*(r2 + 2*x_n**2))
                    y_dist = y_n * radial + (p1*(r2 + 2*y_n**2) + 2*p2*x_n*y_n)

                    u_proj = fx * x_dist + cx
                    v_proj = fy * y_dist + cy
                else:
                    # 无畸变，需要优化
                    u_proj = fx * x_n + cx
                    v_proj = fy * y_n + cy

                # 获取观测到的 2D 坐标
                u_obs, v_obs = frame.get_2d_position(feature_idx)
                
                residuals.append(u_proj - u_obs)
                residuals.append(v_proj - v_obs)

        return np.array(residuals)

    def optimize(self):
        """执行优化"""
        x0 = self.pack_params()

        sparsity = self.get_sparsity_matrix(x0)
        
        # 使用更具鲁棒性的损失函数 (loss='soft_l1') 来抑制外点
        res = scipy.optimize.least_squares(
            self.get_residuals, 
            x0, 
            jac_sparsity=sparsity,
            x_scale='jac',
            method='trf', 
            loss='soft_l1', 
            f_scale=1.0, # 这里的 f_scale 相当于像素误差的阈值
            verbose=2
        )
        
        self.unpack_params(res.x)
        print(f"BA 优化完成。初始误差: {np.linalg.norm(res.fun):.2f}") 

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
                # 1. 关联 3D 点
                sparsity[res_idx : res_idx + 2, p_start : p_start + 3] = 1
                
                # 2. 关联 相机位姿 (非第0帧)
                if frame_idx in self.frame_2_params_idx:
                    f_start = self.frame_2_params_idx[frame_idx]
                    sparsity[res_idx : res_idx + 2, f_start : f_start + 6] = 1
                
                # 3. 关联 相机内参 (重点：如果开启了内参优化，每一行残差都要对内参列置 1)
                if self.intrisic_2_params_idx is not None:
                    # 索引区间从 intrisic_2_params_idx 直到末尾
                    sparsity[res_idx : res_idx + 2, self.intrisic_2_params_idx : ] = 1
                    
                res_idx += 2
                
        return sparsity 
    
    def calculate_rmse(self, params=None):
        """
        计算当前参数下的均方根重投影误差 (Pixels)
        """
        if params is None:
            # 如果不传参，默认计算优化后的当前状态
            residuals = self.get_residuals(self.pack_params())
        else:
            residuals = self.get_residuals(params)

        # residuals 是 [u1-u1', v1-v1', u2-u2', v2-v2', ...]
        # 我们需要每两个一组计算欧式距离，或者直接对整个向量求均方根
        num_observations = len(residuals) / 2 # 每个点有两个坐标观测
        
        # 物理意义上的 RMSE (单位：像素)
        mse = np.mean(np.square(residuals))
        rmse = np.sqrt(mse)
        
        # 计算平均重投影误差 (更直观的 L1 范数)
        # 将 residuals 重新排列为 (N, 2)
        res_reshaped = residuals.reshape(-1, 2)
        l2_norms = np.linalg.norm(res_reshaped, axis=1) # 每个观测点的像素偏差
        mean_error = np.mean(l2_norms)
        max_error = np.max(l2_norms)

        print("-" * 30)
        print(f"BA 质量评估报告:")
        print(f"观测点总数: {int(num_observations)}")
        print(f"RMSE (像素): {rmse:.4f}")
        print(f"平均误差 (像素): {mean_error:.4f}")
        print(f"最大偏差 (像素): {max_error:.4f}")
        print("-" * 30)
        
        return rmse             

                

    



