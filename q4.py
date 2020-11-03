import cv2
import numpy as np
from math import sqrt
import imutils
img_prefix = 'Dataset_opencvdl/Q4_Image/'

def transform(t1, t2, t3, t4):
    cv2.namedWindow("output")
    cv2.resizeWindow("output", 1000, 1000) 
    cv2.moveWindow("output", 20, 20)
    img = cv2.imread(img_prefix + 'Parrot.png', 1)
    try:
        rotation = float(t1)
        scaling = float(t2)
        tx = float(t3)
        ty = float(t4)
        M = cv2.getRotationMatrix2D((tx, ty), rotation, scaling)
        new_img = cv2.warpAffine(img, M, (1000, 1000))
        cv2.imshow("output", new_img)  
    except ValueError:
        # Handle the exception
        print('Please enter an float')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
