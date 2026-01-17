import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_planar_pair(image_path):
    # 1. 读取原始图片 (作为 View 1)
    img1 = cv2.imread(image_path)
    if img1 is None:
        print("Error: Image not found.")
        return
    h, w = img1.shape[:2]

    # 2. 定义一个人为设定的单应矩阵 H_gt (Ground Truth)
    # 这里我们模拟一个透视变换：旋转 + 缩放 + 平移 + 透视畸变
    # 稍微给一点透视感 (H[2,0] 和 H[2,1] 不为0)
    H_gt = np.array([
        [0.9, 0.2, 50],   # 旋转+缩放+X平移
        [-0.1, 0.9, 30],  # 旋转+缩放+Y平移
        [0.0001, 0.00005, 1] # 透视项
    ])

    # 3. 生成 View 2 (用 H_gt 变换 img1)
    # warpPerspective 会自动处理像素插值
    img2 = cv2.warpPerspective(img1, H_gt, (w, h))

    # 4. 保存或显示
    cv2.imwrite("synthetic_view1.jpg", img1)
    cv2.imwrite("synthetic_view2.jpg", img2)
    
    print("生成完毕！")
    print("真实 H 矩阵 (Ground Truth):\n", H_gt)
    
    # 你现在的任务就是：对这两张图跑你的特征匹配和 findHomography
    # 看看算出来的 H 是不是接近 H_gt (注意 H 只有尺度等价性，通常归一化 H[2,2]=1 后比较)

    plt.subplot(121), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('View 1')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)), plt.title('View 2 (Warped)')
    plt.show()
 
if __name__=="__main__":
    generate_synthetic_planar_pair('./data/5.jpg')