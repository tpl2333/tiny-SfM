import cv2
import numpy as np
import matplotlib.pyplot as plt
from model.frame import Frame
from model.camera import Camera

class FeatureMatcher:
    def __init__(self, frame1:Frame,  frame2:Frame, extractor_type='sift', threshold_pixel=3, degeneration=False):

        self.frame1 = frame1
        self.frame2 = frame2
        self.extractor_type = extractor_type
        self.threshold = threshold_pixel
        self.degeneration = degeneration

        self.grayimg1 = cv2.cvtColor(frame1.img, cv2.COLOR_BGR2GRAY)
        self.grayimg2 = cv2.cvtColor(frame2.img, cv2.COLOR_BGR2GRAY)
        
        if self.extractor_type == 'sift':
            self.extractor = cv2.SIFT_create()
            norm_type = cv2.NORM_L2 
        elif self.extractor_type == 'orb':
            self.extractor = cv2.ORB_create()
            norm_type = cv2.NORM_HAMMING 
        else:
            print("[matching] Extractor selection error! Defaulting to SIFT")
            self.extractor = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
        
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)


    def extracting(self):
        """
        修改后的提取函数：
        1. 检查 Frame 是否已有特征点（避免重复计算）
        2. 如果没有，计算并存入 Frame
        """
        # 处理 Frame 1
        if self.frame1.kps is None or self.frame1.des is None:
            kp1, des1 = self.extractor.detectAndCompute(self.grayimg1, None)
            self.frame1.kps = kp1  
            self.frame1.des = des1
        else:
            kp1, des1 = self.frame1.kps, self.frame1.des

        # 处理 Frame 2
        if self.frame2.kps is None or self.frame2.des is None:
            kp2, des2 = self.extractor.detectAndCompute(self.grayimg2, None)
            self.frame2.kps = kp2
            self.frame2.des = des2
        else:
            kp2, des2 = self.frame2.kps, self.frame2.des

                # class KeyPoint:
                #     def __init__(self):
                #         self.pt = (0.0, 0.0)        # 关键点的(x, y)坐标
                #         self.size = 0.0             # 关键点的直径（特征区域大小）
                #         self.angle = -1.0           # 关键点的方向（角度，-1表示未指定）
                #         self.response = 0.0         # 关键点的响应强度（越大表示越好）
                #         self.octave = 0             # 关键点所在的金字塔层数
                #         self.class_id = -1          # 关键点的类别ID
                # kp 是KeyPoints列表
                
                # descriptor是一个NumPy数组，其形状为 (n_keypoints, descriptor_dimension)


    def matching(self):
        """
        kp1, kp2: 关键点数组
        des1, des2: 特征描述子数组
        
        return: 
            F/H: 基本或单应矩阵
            inliermatches: 内点匹配点
            model_type: "F" or "H"
        """

        if self.frame1.des is None or self.frame2.des is None or len(self.frame1.des) < 2 or len(self.frame2.des) < 2:
            print("the number of keypoints less than 2")
            return None, []
    
        #1、knn match
        raw_matches = self.matcher.knnMatch(self.frame1.des, self.frame2.des, k=2)

                # class DMatch:
                #     def __init__(self):
                #         self.queryIdx = -1      # 查询图像中描述符的索引
                #         self.trainIdx = -1      # 训练图像中描述符的索引
                #         self.imgIdx = -1        # 训练图像的索引（用于多图像匹配）
                #         self.distance = 0.0     # 两个描述符之间的距离

                # Match返回DMatch列表，knnMatch返回二维DMatch列表，例如当k=2时返回
                # [
                #     [DMatch1, DMatch2],  # 第1个查询描述符的2个最佳匹配
                #     [DMatch1, DMatch2],  # 第2个查询描述符的2个最佳匹配
                #     ...,
                #     [DMatch1, DMatch2]   # 第N个查询描述符的2个最佳匹配
                # ]

        #2.lowe's ratio test
        good_matches = []
        for m, n in raw_matches:
            if m.distance< 0.75*n.distance:
                good_matches.append(m)

        #3.RANSAC geometry verification  
        #print("good_matches_num", len(good_matches))      
        if len(good_matches)>8:
            
            # 提取匹配点对，并转换为[N,1,2]的numpy格式
            pts1 = np.float32([self.frame1.kps[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([self.frame2.kps[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算基础矩阵
            # method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
            method = cv2.RANSAC

            if not self.degeneration: 
            #认为完全不会退化，选择基本矩阵模型
                F, mask = cv2.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=0.999)

                    #mask是N*1的numpy数组，1表示内点（符合基础矩阵），0表示外点（不符合）

                # 筛选内点
                matches_mask = mask.ravel().tolist()
                inlier_matches=[]
                for i, match in enumerate(good_matches):
                    if matches_mask[i]==1:
                        inlier_matches.append(match)
                
                return F, inlier_matches, "F"
            else:  
            #Homography vs. Fundamentdal
                
                # 3.1 分别计算误差
                # Homography 对称转移误差
                H, mask_H = cv2.findHomography(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=0.999)

                if H is None:
                    GRIC_H = float("inf")
                else:
                    H_inv = np.linalg.inv(H)

                    pts1_proj = cv2.perspectiveTransform(pts1, H).reshape(-1,2)
                    pts2_proj = cv2.perspectiveTransform(pts2, H_inv).reshape(-1,2)

                    error_fwd = np.sum((pts2.reshape(-1,2) - pts1_proj)**2, axis=1)
                    error_bwd = np.sum((pts1.reshape(-1,2) - pts2_proj)**2, axis=1)
                    total_errors = (error_fwd + error_bwd)/2

                    GRIC_H = self.calculate_GRIC(total_errors, len(good_matches), model_type="H")

                # Fundamentdal Sampson误差
                F, mask_F = cv2.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=0.999)
                
                if F is None:
                    GRIC_F = float("inf")
                else:
                    x1 = np.hstack((pts1.reshape(-1, 2), np.ones((pts1.shape[0], 1))))  #齐次化(N,3)
                    x2 = np.hstack((pts2.reshape(-1, 2), np.ones((pts2.shape[0], 1))))

                    Fx1 = np.dot(F, x1.T).T 
                    FTx2 = np.dot(F.T, x2.T).T

                    xfx = np.sum(x2 * Fx1, axis=1)
                    denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + FTx2[:, 0]**2 + FTx2[:, 1]**2
                    sampson_errors = (xfx**2) / (denom + 1e-8)

                    GRIC_F = self.calculate_GRIC(sampson_errors, len(good_matches), model_type="F")

                #3.2 GRIC选择模型
                inliers_H_num = np.sum(mask_H)
                inliers_F_num = np.sum(mask_F)
                
                # 如果 H 的内点数接近 F 的内点数 (例如 > 80%)，强制选择 H 防止平面场景被过拟合为 F
                ratio = inliers_H_num / (inliers_F_num + 1e-8)
                
                print(f"GRIC_H: {GRIC_H:.2f}, GRIC_F: {GRIC_F:.2f}")
                print(f"Inliers H: {inliers_H_num}, Inliers F: {inliers_F_num}, Ratio: {ratio:.2f}")

                # 如果 GRIC 倾向于 H，或者 H 的内点比率足够高，就选 H
                if GRIC_H < GRIC_F or ratio > 0.8: 
                    print("Select H (Planar/Rotation)")
                    matches_mask = mask_H.ravel().tolist()
                    inlier_matches=[]
                    for i, match in enumerate(good_matches):
                        if matches_mask[i]==1:
                            inlier_matches.append(match)
                    return H, inlier_matches, "H"
                else:
                    print("Select F (General 3D)")
                    matches_mask = mask_F.ravel().tolist()
                    inlier_matches=[]
                    for i, match in enumerate(good_matches):
                        if matches_mask[i]==1:
                            inlier_matches.append(match)
                    return F, inlier_matches, "F"
        else:
            print("the number of good_matches less than 8")
            return None, []

        
    def draw_matches(self, matches):
        
        img_match = cv2.drawMatches(self.frame1.img, self.frame1.kps, self.frame2.img, self.frame2.kps, matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Matches: {len(matches)}")
        plt.axis('off')
        plt.show()

    def calculate_GRIC(self, residuals, N, model_type):
        """
        residuals: 每一个匹配点的残差数组 (未求和)
        N: 所有匹配点的总数 (len(good_matches))
        model_type: 'H' or 'F'
        """
        
        # 设定常数
        # 这里的 lambda 对应公式中的惩罚系数
        # lambda_1 * d * N + lambda_2 * k
        lambda_1 = 2.0 
        lambda_2 = np.log(4) 
        
        # 截断阈值 T 
        T = 9
        
        # 1. 计算第一项：鲁棒残差和
        robust_residuals = np.minimum(residuals, T) 
        sum_residuals = np.sum(robust_residuals)
        
        # 2. 设定 k 和 d
        if model_type == 'F':
            k = 7  # F 有 7 个自由度
            d = 3  # F 把 4D 数据压缩到 3D 流形 (1个约束)
        elif model_type == 'H':
            k = 8  # H 有 8 个自由度
            d = 2  # H 把 4D 数据压缩到 2D 流形 (2个约束)
        else:
            raise ValueError("Unknown model type")
            
        # 3. 计算最终 GRIC
        gric = sum_residuals + lambda_1 * d * N + lambda_2 * k
        
        return gric
    
    def forward(self, frame1:Frame, frame2:Frame,):

        matcher.extracting()
        M, final_matches, model_type = matcher.matching()

        return M, final_matches, model_type
    
if __name__=="__main__":

    path1 = "./data/1.png" 
    path2 = "./data/2.png"
   
    cam = Camera() 

    try:
        # 1. 实例化 Frame
        f1 = Frame(path1, cam)
        f2 = Frame(path2, cam)

        h, w, c = f1.img.shape
        cam.set_size(h, w)
        cam.setup_by_guess()

        # 2. 匹配
        matcher = FeatureMatcher(f1, f2, extractor_type='sift', degeneration=True)
        
        # 3. 提取 (此时会自动把特征存入 f1 和 f2)
        matcher.extracting()
        
        # 4. 匹配 
        M, final_matches, model_type = matcher.matching()
        
        if M is not None:
            if model_type == "F":
                print("\n基础矩阵 F:\n", M)
                matcher.draw_matches(final_matches)   
            elif model_type == "H":
                print("\n单应矩阵 H:\n", M)
                matcher.draw_matches(final_matches)
            else:
                raise ValueError(f"[FeatureMatcher] model_type error!") 
        else:
            raise ValueError(f"[FeatureMatcher] Output matrix is None!")

    except Exception as e:
        print(f"发生错误: {e}")



    