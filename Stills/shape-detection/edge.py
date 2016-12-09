import cv2
import numpy as np

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(drawing,[cnt],0,(255,255,255),3)
        cv2.CHAIN_APPROX_SIMPLE
        cv2.imshow('output',drawing)
    return drawing

img = cv2.imread('patrat.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = 100
max_thresh = 255
drawing = thresh_callback(thresh)
cv2.imshow('im',drawing)
cv2.imwrite('corect.png',drawing)
print drawing
if cv2.waitKey(0) == 27:
	cv2.destroyAllWindows()