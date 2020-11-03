import cv2
import numpy as np

img_prefix = 'Dataset_opencvdl/Q2_Image/'

def median_filter():
    img = cv2.imread(img_prefix + 'Cat.png', 1)
    median = cv2.medianBlur(img, 7)
    cv2.imshow("red", median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gaussian_blur():
    img = cv2.imread(img_prefix + 'Cat.png', 1)
    median = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("red", median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def bilateral_filter():
    img = cv2.imread(img_prefix + 'Cat.png', 1)
    median = cv2.bilateralFilter(img, 9, 90, 90)
    cv2.imshow("red", median)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_trackbar(x):
    print(x)
    alpha = x/100
    beta = 1 - alpha
    cv2.addWeighted(img_resize,alpha,img_flip,beta,0.0,dst)
    cv2.imshow("image",dst)

def blending():
    # Create a black image, a window
    cv2.namedWindow('image')
    cv2.imshow("image",img_flip)
    # bar_max = 100
    # create trackbars for color change
    cv2.createTrackbar('Blend','image',0, 100, on_trackbar)
    # cv2.imshow("image",dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()