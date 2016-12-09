import cv2
import numpy as np
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

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     cv2.drawContours(drawing,[cnt],0,(255,255,255),2)
    #     cv2.CHAIN_APPROX_SIMPLE
    return contours
cap = cv2.VideoCapture(0)
while(True):

	ret,img = cap.read()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(5,5),0)
	thresh = 100
	max_thresh = 255
	cv2.imshow('original',img)
	contours = thresh_callback(thresh)
	for cnt in contours:
		hull = fill(cnt,img.shape[1],img.shape[0])
		cv2.drawContours(img,[hull],0,(0,255,0),2)
	cv2.imshow('im',img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()