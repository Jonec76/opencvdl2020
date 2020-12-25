import cv2
import numpy as np

img_prefix = 'Datasets/Q1_Image/'


def find_contour():
    img = ['coin01.jpg', 'coin02.jpg']
    for i in range(2):
        image = cv2.imread(img_prefix + img[i], 1)
        image = cv2.GaussianBlur(image,(5,5),0)
        # Grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        # Find Canny edges 
        edged = cv2.Canny(gray, 30, 200) 
        
        # Finding Contours 
        # Use a copy of the image e.g. edged.copy() 
        # since findContours alters the image 
        contours, hierarchy = cv2.findContours(edged,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            
        # Draw all contours 
        # -1 signifies drawing all contours 
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2) 
        
        cv2.imshow(img[i], image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 

    

def count_coins(l1):
    ans = []
    img = ['coin01.jpg', 'coin02.jpg']
    for i in range(2):
        image = cv2.imread(img_prefix + img[i], 1)
        image = cv2.GaussianBlur(image,(5,5),0)
        # Grayscale 
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        
        # Find Canny edges 
        edged = cv2.Canny(gray, 30, 200) 
        
        # Finding Contours 
        # Use a copy of the image e.g. edged.copy() 
        # since findContours alters the image 
        contours, hierarchy = cv2.findContours(edged,  
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        ans.append(len(contours))
    
    ans_str = "There are %d conis in coin01.jpg\nThere are %d conis in coin02.jpg" %(ans[0], ans[1])
    l1.setText(ans_str)
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
    


