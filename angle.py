# function:
import cv2
import numpy as np
import os
def angle(img):
    # path = "./00000.png"
    # img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(352,160))
    show_img = img.copy()
    rows, cols = img.shape[:2]
    img = img[85:rows,0:cols]
    cv2.namedWindow("img",0)

    mask_global = np.where((img == 1) | (img == 4)| (img == 0), 0, 1).astype('uint8')
    # kernel = np.ones((1, 1), np.uint8)
    # mask_global = cv2.erode(mask_global, kernel, iterations=2)
    mask_global = cv2.medianBlur(mask_global, 3, 4)
    _,contours, hierarchy = cv2.findContours(mask_global, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sh_cont = np.shape(contours)
    print(np.shape(sh_cont))
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    if sh_cont[0]==0 or np.shape(sh_cont)[0]>3:
        print("none")
        cha = 0
        cv2.imshow("img_no", show_img*50)
        cv2.imshow("mask_global", mask_global * 50)
        cv2.waitKey(0)
        return cha
    else:
        rows,cols = img.shape[:2]
        img_back = np.zeros(np.shape(img))
        angle = []
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.uint(box)
            mask_global = cv2.drawContours(mask_global*10, [box], 0, (255,255,255), 2)
            angle.append(rect[2])
        # [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_HUBER, 0, 0.01, 0.01)
        # lefty = int((-x * vy / vx) + y)
        # k = vy/vx
        # angle = np.arctan(k) + np.pi/2
        # print(angle)
        # righty = int(((cols - x) * vy / vx) + y)
        # img_back = cv2.line(img_back, (cols - 1, righty), (0, lefty), (255,255,255), 2)
        # print(angle)

        cha = -angle[0]-(90+angle[1])
        cv2.imshow("img",mask_global)
        cv2.waitKey(0)
        return cha
        # cv2.imshow("img",img_back)

for fpathe,dirs,fs in os.walk('./222/'):
  for f in fs:
    # print(os.path.join(fpathe,f))
    imgpath = os.path.join(fpathe,f)
    img = cv2.imread(imgpath)
    angle(img)
    print(imgpath)


