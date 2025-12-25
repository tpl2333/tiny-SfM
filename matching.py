import cv2
import numpy as np
import matplotlib.pyplot as plt

class FeatureMatcher:
    def __init__(self, imgpath1, imgpath2, extractor_type='sift'):
        self.imgpath1 = imgpath1
        self.imgpath2 = imgpath2
        self.extractor_type=extractor_type

        self.img1 = cv2.imread(imgpath1)
        self.img2 = cv2.imread(imgpath2)
        if self.img1 is not None and self.img2 is not None:
            self.grayimg1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
            self.grayimg2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError(f"Image path error! Cannot read {imgpath1} or {imgpath2}")
        
        if self.extractor_type == 'sift':
            self.extractor = cv2.SIFT_create()
            norm_type = cv2.NORM_L2 
        elif self.extractor_type == 'orb':
            self.extractor = cv2.ORB_create()
            norm_type = cv2.NORM_HAMMING 
        else:
            print("Extractor selection error! Defaulting to SIFT")
            self.extractor = cv2.SIFT_create()
            norm_type = cv2.NORM_L2
        
        self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)


    def extracting(self):

        kp1, des1 = self.extractor.detectAndCompute(self.grayimg1, None)
        kp2, des2 = self.extractor.detectAndCompute(self.grayimg2, None)

                # class KeyPoint:
                #     def __init__(self):
                #         self.pt = (0.0, 0.0)        # 关键点的(x, y)坐标
                #         self.size = 0.0             # 关键点的直径（特征区域大小）
                #         self.angle = -1.0           # 关键点的方向（角度，-1表示未指定）
                #         self.response = 0.0         # 关键点的响应强度（越大表示越好）
                #         self.octave = 0             # 关键点所在的金字塔层数
                #         self.class_id = -1          # 关键点的类别ID

                # descriptor是一个NumPy数组，其形状为 (n_keypoints, descriptor_dimension)

        return kp1, des1, kp2, des2
    
    def matching(self, kp1, des1, kp2, des2):

        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            print("the number of keypoints less than 2")
            return None, []
    
        #1、knn match
        raw_matches = self.matcher.knnMatch(des1, des2, k=2)

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
        if len(good_matches)>8:

            # 提取匹配点对，并转换为[N,1,2]的numpy格式
            pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # 使用RANSAC计算基础矩阵
            # method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
            method = cv2.RANSAC
            F, mask = cv2.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=1.0, confidence=0.999)

                    #mask是N*1的numpy数组，1表示内点（符合基础矩阵），0表示外点（不符合）

            # 筛选内点
            matches_mask = mask.ravel().tolist()
            inlier_matches=[]
            for i, match in enumerate(good_matches):
                if matches_mask[i]==1:
                    inlier_matches.append(match)
            
            return F, inlier_matches
        else:
            print("the number of good_matches less than 8")
            return None, []
        
    def draw_matches(self, kp1, kp2, matches):
        
        img_match = cv2.drawMatches(self.img1, kp1, self.img2, kp2, matches, None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
        plt.title(f"Final Matches: {len(matches)}")
        plt.axis('off')
        plt.show()

if __name__=="__main__":

    path1 = "./data/7.jpg" 
    path2 = "./data/8.jpg"

    try:
        matcher = FeatureMatcher(path1, path2, extractor_type='sift')
        kp1, des1, kp2, des2 = matcher.extracting()
        F, final_matches = matcher.matching(kp1, des1, kp2, des2)
        if F is not None:
            print("\n基础矩阵 F:\n", F)
            matcher.draw_matches(kp1, kp2, final_matches)
    except Exception as e:
        print(f"发生错误: {e}")
        print("提示：请确保目录下有 1.jpg 和 2.jpg，或者修改代码中的 path1, path2")



    