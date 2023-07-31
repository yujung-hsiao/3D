import cv2
import numpy as np
import matplotlib.pyplot as plt


def pre(img, e1 = 5, d = 3, e2 = 1):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(10, 10))
    equalized_clahe = clahe.apply(img)
    binary_img = cv2.threshold(equalized_clahe, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    kernel_e = np.ones((e1, e1), np.uint8)
    kernel_d = np.ones((d, d), np.uint8)
    kernel_e2 = np.ones((e2, e2), np.uint8)
    erosion = cv2.erode(binary_img, kernel_e, iterations=1)
    dilation = cv2.dilate(erosion, kernel_d, iterations=1)
    a = cv2.erode(dilation, kernel_e2, iterations=1)
    return a

def draw_histogram(img):
    plt.hist(img.ravel(), 256)
    plt.show()

def split(img):
    length, width = img.shape
    img1 = img[:length//3, :width//3]
    img2 = img[:length//3, width//3: 2*width//3]
    img3 = img[:length//3, 2*width//3:]
    img4 = img[length//3:2*length//3, :width//3]
    img5 = img[length//3:2*length//3, width//3: 2*width//3]
    img6 = img[length//3:2*length//3, 2*width//3:]
    img7 = img[2*length//3:, :width//3]
    img8 = img[2*length//3:, width//3: 2*width//3]
    img9 = img[2*length//3:, 2*width//3:]
    top = np.hstack((pre(img1), pre(img2), pre(img3)))
    mid = np.hstack((pre(img4), pre(img5), pre(img6)))
    botton = np.hstack((pre(img7), pre(img8), pre(img9)))
    new_img = np.vstack((top, mid, botton))
    return new_img

# [150:550, 130:550] 
# img = cv2.imread('0302/1.jpg', 0)
# print('blind')
# blind_img = Blind(img)
# draw_histogram(blind_img)

# bg = cv2.imread('0302/bg.jpg', 0) 
# sub = cv2.subtract(blind_img, bg)
# img = cv2.imread('0420/ball/0.jpg', 0)
# cv2.imshow('show', img[100:500, 130:550])
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# cv2.imshow('image', np.hstack((pre(sub_clahe), pre(new_img))))
# cv2.waitKey(0)
# cv2.destroyAllWindows()