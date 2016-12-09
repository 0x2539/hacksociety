import numpy as np
import cv2
import random
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

    contours,h = cv2.findContours(thresh,1,2)  
#print contours
#print type(contours)

    print contours and contours[0]
    print 'hello'
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
    #color = random_color()
       # print len(approx)
        if len(approx)==5:
           # print "pentagon"
            cv2.drawContours(img,[cnt],0,255,-1)
        elif len(approx)==3:
           # print "triangle"
            cv2.drawContours(img,[cnt],0,random_color(),-1)
        elif len(approx)==4:
            #print "square"
            cv2.drawContours(img,[cnt],0,random_color(),-1)
        elif len(approx) == 9:
          #  print "half-circle"
            cv2.drawContours(img,[cnt],0,random_color(),-1)
        elif len(approx) > 15:
           # print "circle"
            cv2.drawContours(img,[cnt],0,random_color(),-1)
    cv2.imshow('img',img)
    cv2.imshow('th',thresh)
    cv2.imshow('gray',gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()