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