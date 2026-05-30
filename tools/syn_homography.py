import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def generate_synthetic_planar_pair(image_path):
    """Generate a planar image pair with a known homography."""
    img1 = cv2.imread(image_path)
    if img1 is None:
        logger.error(f"Image not found: {image_path}")
        return
    h, w = img1.shape[:2]

    # Ground-truth homography with rotation, scale, translation, and perspective.
    H_gt = np.array([
        [0.9, 0.2, 50],
        [-0.1, 0.9, 30],
        [0.0001, 0.00005, 1]
    ])

    img2 = cv2.warpPerspective(img1, H_gt, (w, h))

    cv2.imwrite("synthetic_view1.jpg", img1)
    cv2.imwrite("synthetic_view2.jpg", img2)
    
    logger.info(f"生成完毕。真实 H 矩阵 (Ground Truth):\n{H_gt}")
    
    plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('View 1')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('View 2 (Warped)')
    plt.show()
 
if __name__=="__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    generate_synthetic_planar_pair('./data/5.jpg')
