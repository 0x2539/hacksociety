from cv2 import *
import cv2
import numpy as np
import math
def dist(q,r,f,v):
    return (sqrt((q-r)**2)+(f-v)**2)
def detect(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    dr=0
    for i in range(x,x+w):
        for j in range(len(cnt)):
            dr=dr+dist(cnt[j][0],cnt[j][1],i,y)
    for i in range(y+1,y+h):
        for j in range(len(cnt)):
            dr = dr + dist(cnt[j][0],cnt[j][1],x+w,i)
    for i in range(x,x+w-1):
        for j in range(len(cnt)):
            dr = dr + dist(cnt[j][0],cnt[j][1],i, y+h)
    for i in range(y+1,y+h-1):
        for j in range(len(cnt)):
            dr = dr + dist(cnt[j][0], cnt[j][1], x, i)
    dc=0
    (q, r), rad = cv2.minEnclosingCircle(cnt)
    for i in range(q,q+rad):
        for j in range(len(cnt)):
            dc = dc + dist(cnt[j][0],cnt[0][j],i,r-rad+i)
            dc = dc + dist(cnt[j][0],cnt[0][j],i,r+rad-i)
    for i in range(q-rad,q):
        for j in range(len(cnt)):
            dc = dc + dist(cnt[j][0],cnt[0][j],i,r+i)
            dc = dc + dist(cnt[j][0],cnt[0][j],i,r-i)
    if dc < dr:
        print "cerc"
        cv2.circle(new,(q,r),rad,(255,0,0),3)
        return "cerc"
        
    else:
        print "rectangle"
        cv2.rectangle(new,(x,y),(x+w,y+w),(255,0,0),4)
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
    #cv2.imshow('dst',img)
    return corners

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
    contours = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cv2.drawContours(drawing,[cnt],0,(255,255,255),2)
    #     cv2.CHAIN_APPROX_SIMPLE
    return contours
cam=VideoCapture(0)
s,img=cam.read()
new=np.zeros((480,640,3),np.uint8)
if s:
    img = img[50:400, 100:540]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    cv2.imshow('blur', blur)
    cv2.imshow('img',img)
    print 'hei'
    thresh = 100
    max_thresh = 255
    contours = thresh_callback(thresh)
    print img.shape[0],img.shape[1]
    #for cnt in contours:
        #corners = fill(cnt, img.shape[1], img.shape[0])
        #cv2.drawContours(img, [cnt], 0, (0, 255, 0), 4)
        #cv2.imshow('im', img)
        #shape=detect(cnt)
    #cv2.imshow('output',new)
    cv2.waitKey(0)
cv2.destroyAllWindows()