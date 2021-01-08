import cv2
import numpy as np
import glob, os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
img_prefix = 'Datasets/Q4_Image/'

def onclick(event, depth, disparity):
    _, ax = plt.subplots()
    x = event.x
    y = event.y
    dpth = depth[x, y]
    dis = disparity[x, y]
    txt = "depth: {}\ndisparity: {}\n".format(dpth, dis)
    ax.text(0.7, 0.7, txt , fontsize=14, transform = ax.transAxes, horizontalalignment='right', backgroundcolor='0.75')
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('gray', 800, 800)
    plt.imshow(depth, 'gray')
    plt.show()

def stereo_disparity(self):
    imgL = cv2.imread(img_prefix + 'imgL.png', 0)
    imgR = cv2.imread(img_prefix + 'imgR.png', 0)
    stereo = cv2.StereoSGBM_create(numDisparities=256, blockSize=7)
    depth = stereo.compute(imgL, imgR)
    focal = 178
    baseline = 2826
    fig, ax = plt.subplots()
    disparity = (focal * baseline) // depth
    cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('gray', 800, 800)
    plt.imshow(depth, 'gray')
    cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event, depth, disparity))
    plt.show()