import cv2
import numpy as np
import glob, os
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
img_prefix = 'Datasets/Q3_Image/'

def draw(img, corners, img_points):
    img_points = np.int32(img_points).reshape(-1, 2)
    for i, j in zip(range(3), [1,2,0]):
        img = cv2.line(img, tuple(img_points[i]), tuple(
            img_points[3]), (0, 0, 255), 20)
        img = cv2.line(img, tuple(img_points[i]), tuple(
            img_points[j]), (0, 0, 255), 20)

    return img
    
def augmented_reality():
    with np.load('output.npz') as X:
        mtx, dist, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]


    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_point = np.zeros((11*8, 3), np.float32)
    obj_point[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)

    axis = np.float32([[1, 1, 0], [5, 1, 0],
                        [3, 5, 0], [3, 3, -3]])

    file_path = list()
    for i in range(5):
        path = os.path.join(img_prefix, str(i+1) + '.bmp')
        file_path.append(path)

    ims = list()
    fig = plt.figure('augmented_reality')

    for p in file_path:
        img = cv2.imread(p)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res, corners = cv2.findChessboardCorners(gray, (11, 8), None)
        
        if res == True:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 8), (-1, -1), criteria)
            _, rvecs, tvecs, _= cv2.solvePnPRansac(obj_point, corners2, mtx, dist)
            img_points, _ = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
            
            img = draw(img, corners2, img_points)
            plt_img = img[:, :, ::-1]
            im = plt.imshow(plt_img, animated=True)
            cv2.namedWindow(p, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(p, 1000, 800)
            cv2.imshow(p, plt_img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()