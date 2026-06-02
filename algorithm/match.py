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
    """Feature extraction, pairwise matching, and two-view model selection."""

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
            logger.warning("Unknown extractor type; defaulting to SIFT")
            self.extractor = cv2.SIFT_create()
            norm_type = cv2.NORM_L2

        if self.matcher_type == 'bf':
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)
        else:
            logger.warning("Unknown matcher type; defaulting to BFMatcher")
            self.matcher = cv2.BFMatcher(norm_type, crossCheck=False)
    
    def extract(self, frame:Frame):
        """Extract local features once and cache them on the frame."""
        grayimg = cv2.cvtColor(frame._img, cv2.COLOR_BGR2GRAY)
        if len(frame.kps) < 1:
            kps, des = self.extractor.detectAndCompute(grayimg, None)
            frame.set_feature(kps, des)

    def extract_all(self, frames:list[Frame]):
        """Extract features for a list of frames."""
        if not frames:
            return False
        
        for f in frames:
            self.extract(f)

    
    def match_2d_pair(self, f1:Frame, f2:Frame):
        """Match two frames and select either homography or fundamental geometry.

        Returns the selected model, its inlier matches, the inlier ratio, the
        selected model type ("H" or "F"), and both GRIC scores.
        """

        if len(f1.des) < 8 or len(f2.des) < 8:
            raise InsufficientMatchesError("[match] the number of matching points error, none or less than 8")
        
        raw_matches = self.matcher.knnMatch(f1.des, f2.des, k=2)

        # Lowe ratio test removes ambiguous descriptor matches.
        ratio_matches = []
        for m, n in raw_matches:
            if m.distance< 0.75*n.distance:
                ratio_matches.append(m)

        # Keep the best one-to-one assignment after the ratio test.
        ratio_matches.sort(key=lambda x: x.distance)
        unique_matches = []
        used_q = set()
        used_t = set()
        for m in ratio_matches:
            if m.queryIdx not in used_q and m.trainIdx not in used_t:
                unique_matches.append(m)
                used_q.add(m.queryIdx)
                used_t.add(m.trainIdx)

        pts1 = np.float32([f1.kps[m.queryIdx].pt for m in unique_matches]).reshape(-1, 1, 2)
        pts2 = np.float32([f2.kps[m.trainIdx].pt for m in unique_matches]).reshape(-1, 1, 2)

        # RANSAC keeps the view graph robust to descriptor outliers.
        # method = cv2.USAC_MAGSAC if hasattr(cv2, 'USAC_MAGSAC') else cv2.RANSAC
        method = cv2.RANSAC

        if len(unique_matches) < 10:
            raise InsufficientMatchesError("[match] Too few unique matches after filtering.")

        # Homography score uses symmetric transfer error.
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

        # Fundamental matrix score uses Sampson approximation.
        F, mask_F = cv2.findFundamentalMat(pts1, pts2, method, ransacReprojThreshold=self.threshold, confidence=self.confidence)
        
        if F is None:
            GRIC_F = float("inf")
        else:
            if F.shape[0] > 3:
                F = F[:3, :]

            x1 = np.hstack((pts1.reshape(-1, 2), np.ones((pts1.shape[0], 1))))
            x2 = np.hstack((pts2.reshape(-1, 2), np.ones((pts2.shape[0], 1))))

            Fx1 = np.dot(F, x1.T).T 
            FTx2 = np.dot(F.T, x2.T).T

            xfx = np.sum(x2 * Fx1, axis=1)
            denom = Fx1[:, 0]**2 + Fx1[:, 1]**2 + FTx2[:, 0]**2 + FTx2[:, 1]**2
            sampson_errors = (xfx**2) / (denom + 1e-8)

            GRIC_F = self.calculate_GRIC(sampson_errors, len(unique_matches), model_type="F")

        if mask_H is None and mask_F is None:
            raise InsufficientMatchesError("[match] Geometric verification failed for both H and F.")

        # Prefer H for planar or pure-rotation pairs when it explains nearly
        # the same inliers as F; otherwise keep F for general 3D geometry.
        inliers_H_num = 0 if mask_H is None else np.sum(mask_H)
        inliers_F_num = 0 if mask_F is None else np.sum(mask_F)
        
        HF_ratio = inliers_H_num / (inliers_F_num + 1e-8)
        
        logger.debug(f"GRIC_H: {GRIC_H:.2f}, GRIC_F: {GRIC_F:.2f}")
        logger.debug(f"Inliers H: {inliers_H_num}, Inliers F: {inliers_F_num}, Ratio: {HF_ratio:.2f}")

        if GRIC_H < GRIC_F or HF_ratio > 0.8: 
            if mask_H is None:
                raise InsufficientMatchesError("[match] Homography selected but has no inlier mask.")
            logger.debug("Selected H model (planar or rotation-like pair)")
            matches_mask = mask_H.ravel().tolist()
            model = H
            model_type = "H"
        else:
            if mask_F is None:
                raise InsufficientMatchesError("[match] Fundamental matrix selected but has no inlier mask.")
            logger.debug("Selected F model (general 3D pair)")
            matches_mask = mask_F.ravel().tolist()
            model = F
            model_type = "F"  
        
        inlier_matches=[]
        for i, match in enumerate(unique_matches):
            if matches_mask[i]==1:
                inlier_matches.append(match)
            
        inlier_ratio = len(inlier_matches)/len(unique_matches)
        if not inlier_matches:
            raise InsufficientMatchesError("[match] No inliers after geometric verification.")

        return model, inlier_matches, inlier_ratio, model_type, GRIC_F, GRIC_H

    def calculate_GRIC(self, residuals, N, model_type):
        """Compute a simple GRIC-like model score; lower is better."""
       
        lambda_1 = 2.0 
        lambda_2 = np.log(4) 
        
        T = 9
        
        robust_residuals = np.minimum(residuals, T) 
        sum_residuals = np.sum(robust_residuals)
        
        if model_type == 'F':
            k = 7
            d = 3
        elif model_type == 'H':
            k = 8
            d = 2
        else:
            raise ValueError("Unknown model type")
            
        gric = sum_residuals + lambda_1 * d * N + lambda_2 * k
        
        return gric

    def match_exhaustive(self, frames:list[Frame], viewgraph:ViewGraph):
        """Match every frame pair and insert valid edges into the view graph."""
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
 


    
