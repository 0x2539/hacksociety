from cv2 import *
import cv2
import numpy as np
from math import sqrt
from shape_detector import ShapeDetector
def dist(q,r,f,v):
    return (sqrt((q-f)**2)+(r-v)**2)
def detect(cnt,img):
    x, y, w, h = cv2.boundingRect(cnt)
    #cv2.rectangle(new,(x,y),(x+w,y+h),(255,0,0),3)
    arie_romb = 0
    #print img.shape[0]*img.shape[1]
    #ratie = (w*h)/(img.shape[0]*img.shape[1])
    dr = abs(w) * abs(h)-cv2.contourArea(cnt)
    (q, r), rad = cv2.minEnclosingCircle(cnt)
    dc = 3.14 * rad * rad - cv2.contourArea(cnt)
    arie_romb = (abs(w-x))/2 *(abs(h-y))/2
    arie_romb = arie_romb*4
   # print abs(arie_romb-dr),arie_romb,
    print abs(arie_romb-dr)*(w*h)/(img.shape[0]*img.shape[1])
    if abs(arie_romb-dr)*(w*h)/(img.shape[0]*img.shape[1]) < 100:
        print abs(arie_romb-dr)*(w*h)/(img.shape[0]*img.shape[1])
        print "romb"
        cv2.line(new,(x+w/2,y),(x,y+h/2),(0,255,0),3)
        cv2.line(new,(x,y+h/2),(x+w/2,y+h),(0,255,0),3)
        cv2.line(new,(x+w/2,y),(x+w,y+h/2),(0,255,0),3)
        cv2.line(new,(x+w,y+h/2),(x+w/2,y+h),(0,255,0),3)
        return "romb"
    dr=0
    dr=w*h-cv2.contourArea(cnt)
    (q, r), rad = cv2.minEnclosingCircle(cnt)

    dc=3.14*rad*rad-cv2.contourArea(cnt)
    #print (x+w)*(y+h),cv2.contourArea(cnt)
    q=int(q)
    r=int(r)
    #print dc,abs(dr)
    if dc < abs(dr):
        print "cerc"
        cv2.circle(new,(q,r),int(rad),(255,0,0),3)
        return "cerc"
        
    else:
        if abs(dist(x,y,w,y)) < abs(dist(x,y,x,h)):
            print abs((dist(x,y,x+w,y)))
            if abs(dist(x,y,x+w,y)) <100:
            	print "line"
           	 #print dist(x,y,w,y),dist(x,y,x,h)
            	cv2.line(new,(x,y),(x+w,y+h),(0,0,255),3)
            	return "line"
            elif abs(dist(x,y,x,y+h))<100:
           	print "line"
            	print "b"
            	cv2.line(new,(x,y),(x+w,y+h),(0,0,255),3)
            	return "line"
        print "rectangle" 
        cv2.rectangle(new,(x,y),(x+w,y+h),(0,0,255),3)
        return "rectangle"

def fill(contour,width,height):
    #img = np.ndarray(shape = (width,height),dtype=object)
    #img.fill([0,0,0])
    img = np.zeros((height,width,3),np.uint8)
    #print contour[0][0]
    for i in range(len(contour)):
        img[contour[i][0,1],contour[i][0,0]] = [255,0,0]
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = np.float32(img)
   # print img.shape[1],img.shape[0]
    #cv2.imwrite('im.png',img)
    dst = cv2.cornerHarris(img,2,3,0.04)#4,3,0.2)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    #img[dst>0.01*dst.max()]=[0,0,255]
    (x,y) = np.where(dst>0.01*dst.max())
    corners = np.ndarray(shape = (len(x),2),dtype = int)
    for i in range(len(x)):
        corners[i] = (y[i],x[i])
    
    #print corners
    #hull = cv2.convexHull(corners)
    #cv2.drawContours(img,[hull],0,(0,255,0),-1)
    cv2.imshow('dst',img)
    return corners

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    cv2.imshow('ed',edges)
    drawing = np.ones(img.shape,np.uint8)     # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cv2.drawContours(drawing,[cnt],0,(255,255,255),2)
    #     cv2.CHAIN_APPROX_SIMPLE
    return contours
###################################################################################################
# initialize the camera
cam = VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    #namedWindow("cam-test",CV_WINDOW_AUTOSIZE)
    #imshow("cam-test",img)
    new = np.zeros((480,640,3),np.uint8)
    for i in range(new.shape[0]):
         for j in range(new.shape[1]):
            new[i][j] = 255
    #ret,imgn = cap.read()
    img = img[30:400,80:540]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #for i in range(blur.shape[0]):
     #   for j in range(blur.shape[1]):
      #      if blur[i][j] > 100:
       #         blur[i][j] = 255
        ##       blur[i][j] = 0
    cv2.imshow('bin',blur)
    #blur = cv2.bilateralFilter(blur,5,45,55)
    cv2.imshow('blur',blur)
    thresh = 100
    max_thresh = 255
    #cv2.imshow('original',img)
    contours = thresh_callback(thresh)
    a = [len(c) for c in contours]
    	#print len(contours), a, len(contours)
    print len(contours)
    for cnt in range(len(contours)):
        if cnt % 3 == 0:
        #print cnt
        #if (len(cnt) >50 and len(cnt)<90) or (len(cnt)>160 and len(cnt)<300):
	    corners = fill(contours[cnt],img.shape[1],img.shape[0])
	#cv  2.drawContours(img,[cnt],0,(0,0,0),4)
        #print corners
            shape=detect(contours[cnt],img)
    cv2.imshow('im',img)
    cv2.imshow('out',new)
	#print "salut"
    cv2.waitKey(0)
cv2.destroyAllWindows()
