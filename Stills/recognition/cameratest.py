import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
	ret,img = cap.read()
	img = img[30:400,80:540]
	cv2.imshow('detected circles',img)
	k = cv2.waitKey(30) & 0xff
     	if k == 27:
          	break

cv2.destroyAllWindows()
cap.release()

