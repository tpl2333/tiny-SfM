import os
import json
import numpy as np

def quat2rot(qw, qx, qy, qz):
    """将 COLMAP 的四元数 [qw, qx, qy, qz] 转换为 3x3 旋转矩阵 (W2C)"""
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R

def umeyama_alignment(X, Y):
    """
    Umeyama 算法: 计算相似变换 (s, R, t) 使得 s * R * X + t 逼近 Y
    X: 估计值 (N, 3), Y: 真值 (N, 3)
    """
    N = X.shape[0]
    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    
    X_c = X - mu_X
    Y_c = Y - mu_Y
    
    sigma_X = (X_c ** 2).sum() / N
    Sigma = (Y_c.T @ X_c) / N
    
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
        
    R_sim = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (sigma_X + 1e-8)
    t_sim = mu_Y - s * R_sim @ mu_X
    
    return s, R_sim, t_sim

def parse_colmap_images(images_txt_path):
    """解析 worldmap 导出的 colmap images.txt 文件"""
    if not os.path.exists(images_txt_path):
        raise FileNotFoundError(f"找不到 COLMAP 位姿文件: {images_txt_path}")
        
    estimates = {}
    is_data_line = False
    
    with open(images_txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if not is_data_line:
                # 这一行是位姿数据
                parts = line.split()
                # 格式: IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                name = parts[9]
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz = map(float, parts[5:8])
                
                R_w2c = quat2rot(qw, qx, qy, qz)
                t_w2c = np.array([tx, ty, tz])
                # 计算相机光心 (世界坐标系)
                c_world = -R_w2c.T @ t_w2c
                
                # 提取纯文件名用于匹配 (例如 ./train/r_0.png -> r_0)
                clean_name = os.path.basename(name).split('.')[0]
                estimates[clean_name] = {'c': c_world, 'R': R_w2c}
                is_data_line = True
            else:
                # 这一行是 2D 点观测数据，跳过
                is_data_line = False
                
    return estimates

def parse_blender_json(json_path):
    """解析 Blender 的 transforms_train.json 文件 (带 OpenCV 轴向翻转修复)"""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"找不到 Blender 真值文件: {json_path}")
        
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    ground_truth = {}
    for frame in data['frames']:
        file_path = frame['file_path']
        clean_name = os.path.basename(file_path).split('.')[0]
        
        # 1. 提取 Blender 原始的 C2W 矩阵 (4x4)
        T_c2w = np.array(frame['transform_matrix'])
        R_blender_c2w = T_c2w[:3, :3]
        c_gt = T_c2w[:3, 3] # 光心坐标，保持原样
        
        # 2. 【核心修复地雷】：将 Blender (OpenGL: X右, Y上, Z后) 翻转至 OpenCV (COLMAP: X右, Y下, Z前)
        # 相当于给相机系做一个绕 X 轴旋转 180 度的翻转
        R_opencv_c2w = R_blender_c2w @ np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])
        
        # 3. 转换成评估所需的 W2C 矩阵
        R_gt_w2c = R_opencv_c2w.T
        
        ground_truth[clean_name] = {'c': c_gt, 'R': R_gt_w2c}
        
    return ground_truth

def evaluate(colmap_txt, blender_json):
    print("="*60)
    print(" 开始评估自研 SfM 轨迹精度 (Evaluation Pipeline) ")
    print("="*60)
    
    # 1. 加载数据
    est_data = parse_colmap_images(colmap_txt)
    gt_data = parse_blender_json(blender_json)
    
    # 2. 求交集并匹配
    matched_names = sorted(list(set(est_data.keys()) & set(gt_data.keys())))
    num_matched = len(matched_names)
    print(f"-> 成功匹配的相机帧数: {num_matched} / 输入总帧数: {len(gt_data)}")
    
    if num_matched < 3:
        print("错误: 匹配到的帧数太少，无法进行轨迹对齐评测！")
        return
        
    X_est = np.array([est_data[name]['c'] for name in matched_names])
    Y_gt = np.array([gt_data[name]['c'] for name in matched_names])
    
    # 3. 运行 Umeyama 算法进行 Sim(3) 轨迹对齐
    s, R_sim, t_sim = umeyama_alignment(X_est, Y_gt)
    X_est_aligned = (s * R_sim @ X_est.T).T + t_sim
    
    # 4. 计算绝对轨迹误差 ATE (Absolute Trajectory Error)
    errors_pos = np.linalg.norm(X_est_aligned - Y_gt, axis=1)
    ate_rmse = np.sqrt(np.mean(errors_pos ** 2))
    ate_mean = np.mean(errors_pos)
    ate_max = np.max(errors_pos)
    
    # 5. 计算相对旋转误差 RPE (Relative Pose Error for Rotation)
    # 相对旋转角度完全不受全局坐标轴定义和尺度变化的影响
    rot_errors = []
    for i in range(num_matched - 1):
        name_i = matched_names[i]
        name_j = matched_names[i+1]
        
        # 估计值的相对旋转 (i -> j)
        R_est_i = est_data[name_i]['R']
        R_est_j = est_data[name_j]['R']
        delta_R_est = R_est_j @ R_est_i.T
        
        # 真值的相对旋转 (i -> j)
        R_gt_i = gt_data[name_i]['R']
        R_gt_j = gt_data[name_j]['R']
        delta_R_gt = R_gt_j @ R_gt_i.T
        
        # 计算相对旋转的残差矩阵
        E_rot = delta_R_est @ delta_R_gt.T
        
        # 矩阵迹转角度
        trace_val = np.trace(E_rot)
        # 防数值溢出
        trace_val = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
        angle_err = np.arccos(trace_val) * 180.0 / np.pi
        rot_errors.append(angle_err)
        
    rpe_rot_mean = np.mean(rot_errors)
    rpe_rot_max = np.max(rot_errors)
    
    # 6. 打印定量评测报告
    print("\n" + "-"*20 + " 定量评估结果报告 (Quantitative Results) " + "-"*20)
    print(f"【平移指标 - 绝对轨迹误差 ATE】")
    print(f" -> ATE RMSE (均方根误差):   {ate_rmse:.5f} 米")
    print(f" -> ATE Mean (平均绝对误差): {ate_mean:.5f} 米")
    print(f" -> ATE Max  (最大漂移误差): {ate_max:.5f} 米")
    print(f"\n【旋转指标 - 邻帧相对旋转误差 RPE】")
    print(f" -> RPE Rot Mean (平均角度误差): {rpe_rot_mean:.3f} 度")
    print(f" -> RPE Rot Max  (最大角度误差): {rpe_rot_max:.3f} 度")
    print("-"*60)

if __name__ == "__main__":
    # ------------------ 请修改以下路径 ------------------
    # 你的 worldmap.py 导出的目录下的 images.txt 路径
    COLMAP_IMAGES_TXT = "./output/synthetic/ship/images.txt" 
    # Blender Lego 数据集的真值 json 路径
    BLENDER_TRANSFORMS_JSON = "./data/synthetic/ship/transforms_test.json"
    # --------------------------------------------------
    
    evaluate(COLMAP_IMAGES_TXT, BLENDER_TRANSFORMS_JSON)