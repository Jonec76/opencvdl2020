import cv2
import numpy as np

img_prefix = 'Dataset_opencvdl/Q1_Image/'

img = cv2.imread(img_prefix + 'Uncle_Roger.jpg', 1)
height = img.shape[0]
width = img.shape[1]
dst = np.ones((width, height,3), np.uint8)*255
img_flip = cv2.flip(img, 1)
dst = cv2.resize(img, (width, height))

def load_image():
    img = cv2.imread(img_prefix + 'Uncle_Roger.jpg', 1)
    
    print("Height: "+ str(height))
    print("Width: "+str(width))
    # img = cv2.resize(img, (680, 373))                    # Resize image
    # print(img.shape)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_seperation():
    img = cv2.imread(img_prefix + 'Flower.jpg', 1)

    # Method 1: copy image and set other channels to black
    r = img.copy()
    r[:,:,0] = r[:,:,1] = 0

    g = img.copy()
    g[:,:,0] = g[:,:,2] = 0

    b = img.copy()
    b[:,:,1] = b[:,:,2] = 0

    cv2.imshow("red",r)
    cv2.imshow("green",g)
    cv2.imshow("blue",b)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_flipping():
    img = cv2.imread(img_prefix + 'Uncle_Roger.jpg', 1)
    # img = cv2.resize(img, (680, 373))                    # Resize image

    img_flip = cv2.flip(img, 1)
    cv2.imshow('image',img)
    cv2.imshow('flip image',img_flip)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def on_trackbar(x):
    alpha = x/100
    beta = 1 - alpha
    cv2.addWeighted(img,alpha,img_flip,beta,0.0,dst)
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