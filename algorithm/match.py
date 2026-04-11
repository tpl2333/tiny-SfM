import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
from tqdm import tqdm

from model.frame import Frame
from model.edge import EdgeData
from management.viewgraph import ViewGraph
from algorithm.errors import *

class FeatureMatcher:
    def __init__(self, extractor_type='sift', matcher_type='bf',threshold_pixel=3, confidence=0.999):

        self.extractor_type = extractor_type
        self.matcher_type = matcher_type
        self.threshold = threshold_pixel
        self.confidence = confidence
   
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

        if self.matcher_type == 'bf':
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        else:
            print("[matching] Matcher selection error! Defaulting to BFMatcher")
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    
    # --------- 核心方法 ---------
    def extract(self, frame:Frame):
        """
        提取特征方法：
        1. 检查 Frame 是否已有特征点（避免重复计算）
        2. 如果没有，计算并存入 Frame
        """
        grayimg = cv2.cvtColor(frame._img, cv2.COLOR_BGR2GRAY)
        if len(frame.kps) < 1:
            kps, des = self.extractor.detectAndCompute(grayimg, None)
            frame.set_feature(kps, des)

    def extract_all(self, frames:list[Frame]):
        """
        批量提取特征
        """
        if not frames:
            return False
        
        for f in frames:
            self.extract(f)

    
    def match_2d_pair(self, f1:Frame, f2:Frame):
        """        
        两帧之间特征点的匹配与几何验证
        frame1, frame2: 匹配的两帧

        Args:
            f1 (Frame): query_frame
            f2 (Frame): train_frame

        Raises:
            InsufficientMatchesError: 错误：没有足够匹配点

        Returns:
            model, inlier_matches, inlier_ratio, model_type, GRIC_F, GRIC_H
        """

        if len(f1.des) < 8 or len(f2.des) < 8:
            raise InsufficientMatchesError("[match] the number of matching points error, none or less than 8")
        
        # 1. knn 匹配
        raw_matches = self.matcher.knnMatch(f1.des, f2.des, k=2)

        # 2. lowe比率测试，滤除自相似的纹理
        ratio_matches = []
        for m, n in raw_matches:
            if m.distance< 0.75*n.distance:
                ratio_matches.append(m)

        # 3. 双射过滤，避免多对一匹配情况 (A->B，C->B)
        ratio_matches.sort(key=lambda x: x.distance)
        unique_matches = []
        used_q = set()
        used_t = set()
        for m in ratio_matches:
            if m.queryIdx not in used_q and m.trainIdx not in used_t:
                unique_matches.append(m)
                used_q.add(m.queryIdx)
                used_t.add(m.trainIdx)

        # 4. 对极几何验证 
        # 提取匹配点对，并转换为[N,1,2]的numpy格式
        pts1 = np.float32([f1.kps[m.queryIdx].pt for m in unique_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([f2.kps[m.trainIdx].pt for m in unique_matches]).reshape(-1, 1, 2)

        # 使用RANSAC计算基础矩阵
        # method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
        method = cv2.RANSAC

        # 4.1 分别计算误差
        # Homography 对称转移误差
        H, mask_H = cv2.findHomography(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=self.confidence)

        if H is None:
            GRIC_H = float("inf")
        else:
            H_inv = np.linalg.inv(H)

            pts1_proj = cv2.perspectiveTransform(pts1, H).reshape(-1,2)
            pts2_proj = cv2.perspectiveTransform(pts2, H_inv).reshape(-1,2)

            error_fwd = np.sum((pts2.reshape(-1,2) - pts1_proj)**2, axis=1)
            error_bwd = np.sum((pts1.reshape(-1,2) - pts2_proj)**2, axis=1)
            total_errors = (error_fwd + error_bwd)/2

            GRIC_H = self.calculate_GRIC(total_errors, len(unique_matches), model_type="H")

        # Fundamentdal Sampson误差
        F, mask_F = cv2.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=self.confidence)
        
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

            GRIC_F = self.calculate_GRIC(sampson_errors, len(unique_matches), model_type="F")

        # 4.2 GRIC选择模型
        # 如果 H 的内点数接近 F 的内点数 (例如 > 80%)，强制选择 H 防止平面场景被过拟合为 F
        # 最终选择依靠两个指标 1、 GRIC 的倾向 2、内点比例 都满足才会选择更复杂的 F 模型
        inliers_H_num = np.sum(mask_H)
        inliers_F_num = np.sum(mask_F)
        
        HF_ratio = inliers_H_num / (inliers_F_num + 1e-8)
        
        print(f"GRIC_H: {GRIC_H:.2f}, GRIC_F: {GRIC_F:.2f}")
        print(f"Inliers H: {inliers_H_num}, Inliers F: {inliers_F_num}, Ratio: {HF_ratio:.2f}")

        if GRIC_H < GRIC_F or HF_ratio > 0.8: 
            print("Select H (Planar/Rotation)")
            matches_mask = mask_H.ravel().tolist()
            model = H
            model_type = "H"
        else:
            print("Select F (General 3D)")
            matches_mask = mask_F.ravel().tolist()
            model = F
            model_type = "F"  
        
        inlier_matches=[]
        for i, match in enumerate(unique_matches):
            if matches_mask[i]==1:
                inlier_matches.append(match)
            
        inlier_ratio = len(inlier_matches)/len(unique_matches)

        return model, inlier_matches, inlier_ratio, model_type, GRIC_F, GRIC_H

    def calculate_GRIC(self, residuals, N, model_type):
        """
        residuals: 每一个匹配点的残差数组 (未求和)
        N: 所有匹配点的总数 (len(good_matches))
        model_type: 'H' or 'F'

        return:
            GRIC_score: 越高解释性越不好
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

    # --------- 外部调用层 ---------    
    def match_exhaustive(self, frames:list[Frame], viewgraph:ViewGraph):
        """
        执行全图暴力匹配并填充 ViewGraph
        """
        n = len(frames)

        total_pairs = n * (n - 1) // 2
        logger.info(f"开始暴力匹配，共计 {total_pairs} 对图像...")

        with tqdm(total=total_pairs, desc="Pairwise Matching") as pbar:
            for i in range(n):
                for j in range(i + 1, n):
                    f1, f2 = frames[i], frames[j]
                    
                    try:
                        _, matches, ratio, m_type, g_f, g_h = self.match_2d_pair(f1, f2)
                    
                        edge = EdgeData(matches, ratio, m_type, g_f, g_h)
                        viewgraph.add_edge(f1.idx, f2.idx, edge)

                        logger.debug(f"Edge ({f1.idx}, {f2.idx}) added: {len(matches)} inliers.")
                    
                    except InsufficientMatchesError:

                        logger.debug(f"Edge ({f1.idx}, {f2.idx}) failed: insufficient matches.")
                    
                    pbar.update(1)
 


    