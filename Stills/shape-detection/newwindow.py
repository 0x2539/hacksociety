import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(True):
	ret,img = cap.read()
	new = np.zeros((img.shape[1],img.shape[0],3),np.uint8)
	cv2.rectangle(new,(200,0),(400,128),(0,255,0),3)
	cv2.imshow('new',new)
	print img.shape[0],img.shape[1]
	if cv2.waitKey(1) & 0xFF == ord('q'):
	   break
cap.release()
cv2.destroyAllWindows() 