import cv2
import numpy as np
class figura:
	def __init__(self):
		pass
	def compare(self,img):
		patrat = np.zeros((100,100,3),np.uint8)
		cv2.rectangle(patrat,(20,20),(80,80),(0,255,0),3)
		dreptunghi = np.zeros((100,100,3),np.uint8)
		cv2.rectangle(dreptunghi,(20,30),(80,70),(0,255,0),3)
		cerc = np.zeros((100,100,3),np.uint8)
		cv2.circle(cerc,(50,50), 30, (0,0,255), 3)
		triunghi = np.zeros((100,100,3),np.uint8)
		cv2.line(triunghi,(30,70),(70,70),(0,255,0),3)
		cv2.line(triunghi,(30,70),(50,30),(0,255,0),3)
		cv2.line(triunghi,(50,30),(70,70),(0,255,0),3)
		###########################################
		cv2.drawContours(patrat, img, 0, (255,0,0), 3)
		#cv2.imshow('tri',patrat)
		#cv2.waitKey(0)
		return patrat
		