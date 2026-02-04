import cv2
import numpy as np
import open3d as o3d
#from matching import FeatureMatcher
from model.camera import Camera
from model.frame import Frame
from model.mappoint import Point
from model.worldmap import Map

class Reconstruction:
    def __init__(self, frame1:Frame, frame2:Frame, matches:list, map:Map):

        self.map = map

        self.frame1 = frame1
        self.frame2 = frame2
        # 基础矩阵的内点
        self.F_inlier_matches = matches
        # 恢复位姿通过手性检测的内点（预留）
        self.D_inlier_matches = []
        # 最终恢复得到的3D点，归一化后的非齐次坐标
        self.Points_normalized = None
        self.Points_colors = None

    def recover_pose(self):

        pts1 = np.float32([self.frame1.kps[m.queryIdx].pt for m in self.F_inlier_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([self.frame2.kps[m.trainIdx].pt for m in self.F_inlier_matches]).reshape(-1, 1, 2)
        
        E, mask_E = cv2.findEssentialMat(pts1, pts2,self.frame1.camera.K, method=cv2.RANSAC, threshold = 3, prob=0.999)
        retval, R, t, mask_Depth = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.frame1.camera.K, mask = mask_E)

        self.frame2.set_pose(R, t)

        matches_mask = mask_Depth.ravel().tolist()
        for i, match in enumerate(self.F_inlier_matches):
            if matches_mask[i]==1:
                self.D_inlier_matches.append(match)
    
    def triangulation(self):

        P1 = self.frame1.get_proj_matrix()
        P2 = self.frame2.get_proj_matrix()

        pts1 = np.float32([self.frame1.kps[m.queryIdx].pt for m in self.D_inlier_matches]).reshape(-1, 2).T
        pts2 = np.float32([self.frame2.kps[m.trainIdx].pt for m in self.D_inlier_matches]).reshape(-1, 2).T

        Points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
        # Points4D的形状为[4, N] 

        if Points4D is not None:
            self.map.add_frame(self.frame1)
            self.map.add_frame(self.frame2)

            xyz = Points4D[:3, :] #[3, N]
            w = Points4D[3:, :]   #[1, N]
            Points_normalized = (xyz / w).T #归一化并转置 [N, 3]

            self.Points_normalized = Points_normalized

            self.register()
        else:
            print("三角化失败，Point4D为None")

    def register(self):

        assert len(self.D_inlier_matches)==len(self.Points_normalized), "三角化失败，深度内点与实际三角化点数量不一致"

        if not self.frame1.is_registered:
            self.map.add_frame(self.frame1)
        if not self.frame2.is_registered:
            self.map.add_frame(self.frame2)
        
        for i, point_coordi in enumerate(self.Points_normalized):

            mappoint = Point(point_coordi)

            idx1 = self.D_inlier_matches[i].queryIdx
            idx2 = self.D_inlier_matches[i].trainIdx

            mappoint.add_observation(self.frame1, idx1)
            mappoint.add_observation(self.frame2, idx2)

            self.frame1.add_points(mappoint.idx,idx1)
            self.frame2.add_points(mappoint.idx,idx2)

            self.map.add_point(mappoint)

    def get_points_color(self):

        colors = []
        img1 = self.frame1.img # BGR
        
        for m in self.D_inlier_matches:
            pt = self.frame1.kps[m.queryIdx].pt
            # 简单取整，防止越界可以加个 clip
            x, y = int(pt[0]), int(pt[1])
            
            # 边界检查
            y = min(max(y, 0), img1.shape[0]-1)
            x = min(max(x, 0), img1.shape[1]-1)
            
            # OpenCV 是 BGR，Open3D 需要 RGB，且范围 [0, 1]
            b, g, r = img1[y, x]
            colors.append([r/255.0, g/255.0, b/255.0])
            
        self.Points_colors = np.array(colors)
    
    def visualize_point_cloud(self, colors=None):

        # 1. 创建一个 PointCloud 对象
        pcd = o3d.geometry.PointCloud()
        
        # 2. 将 numpy 数组赋值给 Open3D 的 points 属性
        pcd.points = o3d.utility.Vector3dVector(self.Points_normalized.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(self.Points_colors)

        # 3. 创建坐标轴辅助看方向 (原点，大小)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])

        # 4. 渲染！
        o3d.visualization.draw_geometries([pcd, coordinate_frame], 
                                        window_name="3D Reconstruction",
                                        width=800, height=600)
        
    def forward(self):

        self.recover_pose()
        self.triangulation()
        self.get_points_color()
        self.visualize_point_cloud()

# if __name__=="__main__":

#     path1 = "./data/1.png" 
#     path2 = "./data/2.png"
#     map = Map()

#     try:

#         cam = Camera()
#         f1 = Frame(path1, cam)
#         f2 = Frame(path2, cam)

#         h, w, c = f1.img.shape
#         cam.set_size(h, w)
#         cam.setup_by_guess()

#         matcher = FeatureMatcher(f1, f2, extractor_type='sift', degeneration=True)
#         matcher.extracting()
#         M, final_matches, model_type = matcher.matching()

#         reconstructor = Reconstruction(f1,f2,final_matches,map)
#         reconstructor.forward()

#         print(f"图片数量：{len(map.frames)}\n")
#         print(f"三维点数量：{len(map.points)}\n")

#     except Exception as e:
#         print(f"发生错误: {e}")