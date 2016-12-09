import numpy as np
import cv2
import random
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
    hull = cv2.convexHull(corners)
    #cv2.drawContours(img,[hull],0,(0,255,0),-1)
    #cv2.imshow('dst',img)
    return hull


def random_color():
    rgbl=[255,0,0]
    #random.shuffle(rgbl)
    return tuple(rgbl)

cap = cv2.VideoCapture(0)
while(True):
    ret,img = cap.read()
    #img = cv2.imread('poze_mari.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (21, 21), 0)
    ret,thresh = cv2.threshold(gray,160,255,1)
    contours,h = cv2.findContours(thresh,3,5) 
    print len(contours) 
    for cnt in contours:
        print 'yes'
        #print img[0]
        #hull = fill(cnt,img.shape[1],img.shape[0])
        #approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
        # if len(approx)==5:
        #     cv2.drawContours(img,[cnt],0,255,-1)
        # elif len(approx)==3:
        #     cv2.drawContours(img,[cnt],0,random_color(),-1)
        # elif len(approx)==4:
        #     cv2.drawContours(img,[cnt],0,random_color(),-1)
        # elif len(approx) == 9: 
        #     cv2.drawContours(img,[cnt],0,random_color(),-1)
        # elif len(approx) > 15:
        #     cv2.drawContours(img,[cnt],0,random_color(),-1)
        #cv2.drawContours(img,[hull],0,(0,255,0),-1)
            #cv2.imshow('th',thresh)
    #cv2.imshow('gray',gray)
    cv2.imshow('img',img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()