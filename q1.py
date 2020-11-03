import cv2

def load_image():
    img = cv2.imread('Dataset_opencvdl/Q1_Image/Uncle_Roger.jpg',1)
    dim = (680, 373)
    # cv2.resize(img, dim)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_seperation():
    print("yooo")
def image_flipping():
    print("yooo")
def blending():
    print("yooo")