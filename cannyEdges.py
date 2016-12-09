from cv2 import *
import cv2
import numpy as np
from math import sqrt
from shape_detector import ShapeDetector
def dist_p(x,y,q,z):
    return (int(sqrt((x-q)**2+(y-z)**2)))

def maxvec(v):
    minv = 10000
    for i in range(len(v)):
        if v[i] < minv:
            minv = v[i]
    return (minv,i)
def newWin(shape,corners,new):
    minx = 100000
    miny = 100000
    maxx = 0
    maxy = 0
    ym = 0
    ymx = 0
    dmax = [0,0,0]
    if shape == "rectangle" or shape == "square":
        #print len(corners)
        for i in range(len(corners)):
            if corners[i][0] < minx:
                minx = corners[i][0]
            if corners[i][0] > maxx:
                maxx = corners[i][0]
            if corners[i][1] > maxy:
                maxy = corners[i][1]
            if corners[i][1] < miny:
                miny = corners[i][1]
    cv2.rectangle(new,(minx,miny),(maxx,maxy),(0,255,0),3)
    if shape == "triangle":
       for i in range(len(corners)-1):
           for j in range(i+1,len(corners)):
               dist=dist_p(corners[i][0],corners[i][1],corners[j][0],corners[j][1])
               if dist > max:
                   max=dist
                    r=i
                    q=j
       cv2.line(new,(corners[r][o],corners[r][1]),(corners[q][0],corners[q][1]),(255.0.0),3)
       for i in range(len(corners)-1):
           if i != r and i !=q:
               dist = dist_p(corners[r][0], corners[r][1], corners[i][0], corners[i][1])
               if  dist > max:
                   max=dist
                   z=i
       cv2.line(new, (corners[r][o], corners[r][1]), (corners[z][0], corners[z][1]), (255.0.0), 3)
       cv2.line(new, (corners[q][o], corners[q][1]), (corners[z][0], corners[z][1]), (255.0.0), 3)





    if shape == "circle":
        for i in range(len(corners)):
            if corners[i][0] < minx:
                minx = corners[i][0]
                ym = corners[i][1]

            if corners[i][0] > maxx:
                maxx = corners[i][0]
                ymx = corners[i][1]
        xc = int((minx+maxx)/2)
        yc = int((ym+ymx)/2)
        r = int(sqrt((xc-minx)**2 + (ym - yc)**2))
        cv2.circle(new,(xc,yc), r, (0,0,255), 3)

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
    #cv2.imshow('dst',img)
    return corners

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
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
    #ret,imgn = cap.read()
    img = img[50:400,100:540]
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    cv2.imshow('blur',blur)
    thresh = 100
    max_thresh = 255
    #cv2.imshow('original',img)
    contours = thresh_callback(thresh)
    a = [len(c) for c in contours]
    	#print len(contours), a, len(contours)
    for cnt in contours:
        #if (len(cnt) >50 and len(cnt)<90) or (len(cnt)>160 and len(cnt)<300):
	   corners = fill(cnt,img.shape[1],img.shape[0])
	   cv2.drawContours(img,[cnt],0,(0,255,0),4)
        #print corners
	   cv2.imshow('im',img)
	   sd=ShapeDetector()
	   shape=sd.detect(cnt)
	   newWin(shape,corners,new)
	   cv2.imshow('out',new)
	#print "salut"
    cv2.waitKey(0)
    cv2.destroyAllWindows()