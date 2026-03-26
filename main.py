import cv2
import numpy as np
import scipy

from model.camera import Camera
from model.frame import Frame
from model.mappoint import Point
from model.worldmap import Map

from algorithm.match import FeatureMatcher
from algorithm.reconstruction import Reconstruction
from algorithm.ba import BA

path1 = "./data/aloeL.jpg" 
path2 = "./data/aloeR.jpg"
map_obj = Map()

try:
    # 1. 基础设置与特征匹配
    cam = Camera(is_dist=False)
    f1 = Frame(path1, cam)
    f2 = Frame(path2, cam)

    h, w, _ = f1.img.shape
    cam.set_size(h, w)
    cam.setup_by_guess(lock_it=True) # 假设焦距并初始化 K

    matcher = FeatureMatcher(f1, f2, extractor_type='sift', degeneration=True)
    matcher.extracting()
    M, final_matches, model_type = matcher.matching()

    # 2. 初始重建 (Initial Estimation)
    # 这一步会通过分解本质矩阵得到 R, t，并进行三角化
    reconstructor = Reconstruction(f1, f2, final_matches, map_obj)
    reconstructor.recover_pose()
    reconstructor.triangulation()
    reconstructor.get_points_color()

    print("-" * 30)
    print(f"初始重建完成:")
    print(f"图片数量: {len(map_obj.frames)}")
    print(f"三维点数量: {len(map_obj.points)}")
    print(f"Camera 初始内参K:\n{f1.camera.K}")
    print(f"Frame 2 初始位置 t:\n{f2.t.flatten()}")
    print(f"Frame 2 初始旋转 R:\n{f2.R}")
    print("-" * 30)

    # 3. Bundle Adjustment 优化
    print("开始 Bundle Adjustment 优化...")
    optimizer = BA(map_obj)
    optimizer.optimize() 
    optimizer.calculate_rmse()

    print("-" * 30)
    print(f"优化后Camera 内参K:\n{f1.camera.K}")
    print(f"优化后Frame 2 位置 t:\n{f2.t.flatten()}")
    print(f"优化后Frame 2 旋转 R:\n{f2.R}")
    
    # 4. 可视化优化后的结果
    # 注意：我们需要手动更新 reconstructor 里的点云数据用于显示
    # 因为 reconstructor.Points_normalized 只是一个副本
    optimized_pts = []
    for pt_idx, pt in map_obj.points.items():
        optimized_pts.append(pt.position)
    
    reconstructor.Points_normalized = np.array(optimized_pts)
    
    print("正在打开可视化窗口...")
    reconstructor.visualize_point_cloud()

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"发生错误: {e}")