import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while(True):
	ret,img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	gray = np.float32(gray)
	dst = cv2.cornerHarris(gray,2,3,0.04)#4,3,0.2)

	#result is dilated for marking the corners, not important
	dst = cv2.dilate(dst,None)

	# Threshold for an optimal value, it may vary depending on the image.
	img[dst>0.01*dst.max()]=[0,0,255]
	(x,y) = np.where(dst>0.01*dst.max())
	corners = np.ndarray(shape = (len(x),2),dtype = int)
	for i in range(len(x)):
		corners[i] = (y[i],x[i])
	print corners
	#hull = cv2.convexHull(corners)
	#cv2.drawContours(img,[hull],0,(0,255,0),-1)
	cv2.imshow('dst',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()