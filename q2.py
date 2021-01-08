import cv2
import numpy as np
import glob

img_prefix = 'Datasets/Q2_Image/'

def find_corners():
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objp = np.zeros((11*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
   
    objpoints = [] 
    imgpoints = [] 
    images = glob.glob(img_prefix+'*.bmp')
    idx = 0
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11,8), None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,8), (-1,-1), criteria)
            imgpoints.append(corners)
            cv2.drawChessboardCorners(img, (11,8), corners2, ret)
            cv2.namedWindow(str(idx), cv2.WINDOW_NORMAL)
            cv2.resizeWindow(str(idx), 1000, 800)
            cv2.imshow(str(idx), img)
            idx += 1
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    np.savez('output.npz', mtx, dist, rvecs, tvecs)
    np.savez('img_obj_points.npz', imgpoints, objpoints)

def find_intrinsic(self):
    with np.load('output.npz') as X:
        mtx, _, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

    print(mtx)

def find_extrinsic(num):
    with np.load('img_obj_points.npz') as X:
        imgpoints, objpoints = [X[i] for i in ('arr_0', 'arr_1')]
    with np.load('output.npz') as X:
        mtx, dist, _, _ = [X[i]
                            for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

    num = int(num) - 1
    retval, rvecs, tvecs = cv2.solvePnP(objpoints[num-1], imgpoints[num-1], mtx, dist)
    dst, _ = cv2.Rodrigues(rvecs)
    extrinsic_mtx = cv2.hconcat([dst, tvecs])
    print(extrinsic_mtx)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_distortion():
    with np.load('output.npz') as X:
        _, dist, _, _ = [X[i] for i in ('arr_0', 'arr_1', 'arr_2', 'arr_3')]

    print(dist)