import cv2

class Match:
    def __init__(self, imgpath1, imgpath2):
        self.imgpath1 = imgpath1
        self.imgpath2 = imgpath2

        self.img1 = cv2.imread("imgpath1")
        self.img2 = cv2.imread("imgpath2")
        self.grayimg1 = cv2.cvtColor(self.img1, cv2.COLOR_BGR2GRAY)
        self.grayimg2 = cv2.cvtColor(self.img2, cv2.COLOR_BGR2GRAY)

    def matching(self):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.grayimg1, None)
        kp2, des2 = sift.detectAndCompute(self.grayimg2, None)

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x:x.distance)

        good_matches = matches[:30]

        return good_matches
    
if __name__=="__main__":
    ababababababba



    