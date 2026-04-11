import os
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path

# 导入你定义的模块
from model.camera import Camera, CameraSource
from management.worldmap import Map
from algorithm.match import FeatureMatcher
from algorithm.reconstruct import Reconstructor

def main():
    # --- 1. 配置参数 ---
    image_dir = "./data/frame/"  # 请确保此目录下有图片
    
    # 假设图像尺寸，如果没有标定，通常取图像中心
    # 建议先读一张图获取真实尺寸
    sample_img_path = next(Path(image_dir).iterdir())
    sample_img = cv2.imread(str(sample_img_path))
    h, w = sample_img.shape[:2]

    # --- 2. 初始化相机内参 (K) ---
    cam = Camera(width=w, height=h)
    cam.setup_by_guess()
    cam.source = CameraSource.GUESS
    print(f"[Main] 相机内参加载完毕，图像尺寸: {w}x{h}")

    # --- 3. 初始化核心系统 ---
    world_map = Map(cam)
    matcher = FeatureMatcher(extractor_type='sift')
    reconstructor = Reconstructor(world_map, matcher, image_dir)

    # --- 4. 执行重建流 ---
    # 注意：确保 reconstruct.py 里的 add_next_frame 已经按上面的逻辑给 point._color 赋值
    reconstructor.run()

    # --- 5. 可视化彩色点云 ---
    draw_colored_map(world_map)

def draw_colored_map(world_map):
    """
    将 Map 中的 3D 点和颜色提取出来，并使用 Open3D 展示
    """
    points_list = []
    colors_list = []

    print(f"[Visualizer] 正在处理 {len(world_map._points)} 个地图点...")

    for pt_idx, pt in world_map._points.items():
        if pt.is_bad:
            continue
        
        # 获取 3D 坐标
        points_list.append(pt.position3d.flatten())
        
        # 获取颜色
        if hasattr(pt, '_color') and pt.color is not None:
            colors_list.append(pt.color)
        else:
            # 如果没颜色，默认给个绿色区分一下
            colors_list.append([0.0, 1.0, 0.0])

    if not points_list:
        print("[Visualizer] 错误：没有可用的 3D 点！")
        return

    # 转换为 numpy 阵列
    np_points = np.array(points_list)
    np_colors = np.array(colors_list)

    # 创建 Open3D 点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points)
    pcd.colors = o3d.utility.Vector3dVector(np_colors)

    geometries = [pcd]
    for f_idx in world_map._registered_ids:
        frame = world_map.get_frame(f_idx)
        cam_model = create_camera_vis(frame, scale=0.2) # 缩放比例根据场景调整
        geometries.append(cam_model)

    print(f"[Visualizer] 渲染中... (已剔除离群点)")
    
    o3d.visualization.draw_geometries(geometries, window_name="MVG 3D Reconstruction")

def create_camera_vis(frame, scale=0.5):
    """
    为指定的 Frame 创建一个四棱锥线框
    scale: 控制相机模型的大小
    """
    # 1. 获取外参
    R = frame._R
    t = frame._t
    
    # 计算相机在世界坐标系下的光心 C = -R^T * t
    C = -R.T @ t
    C = C.flatten()

    # 2. 定义相机空间下的 5 个关键点 (顶点 + 底面 4 个角)
    # 假设底面在 Z=1 的平面上，根据内参大致确定宽高比
    w = scale 
    h = scale * 0.75
    z = scale * 1.5
    
    # 相机坐标系下的点
    pts_c = np.array([
        [0, 0, 0],     # 顶点 0
        [-w, -h, z],   # 左上 1
        [w, -h, z],    # 右上 2
        [w, h, z],     # 右下 3
        [-w, h, z]     # 左下 4
    ])

    # 3. 将点变换到世界坐标系: Pw = R.T @ (Pc - t)
    # 注意：这里的 t 是平移向量，由于 Pc 是在原点定义的，直接套用公式
    pts_w = (R.T @ (pts_c.T - t)).T

    # 4. 定义线段连接顺序 (0-1, 0-2, 0-3, 0-4 为侧棱；1-2, 2-3, 3-4, 4-1 为底边)
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],
        [1, 2], [2, 3], [3, 4], [4, 1]
    ]

    # 5. 创建 Open3D 对象
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pts_w)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    # 设置颜色（已注册帧用红色，当前帧可以用蓝色区分）
    colors = [[1, 0, 0] for _ in range(len(lines))]
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set
if __name__ == "__main__":
    main()