from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np
import math

#calculate angle between current vector line and lane midpoint line
def angle_between(image, x1, x2, y1, y2, x3, x4, y3, y4): 

	slopeMid = (x2-x1)/(y2-y1)
	slopeVect = (x3-x4)/(y3-y4)
	angle = math.atan((slopeVect-slopeMid)/(1+(slopeVect*slopeMid)))
	angle = math.degrees(angle)
	print(angle)
	#angleBetween = 3.14 - abs(angleMid- angleVect)
	#angle = math.degrees(angleBetween)
	print(angle)
	draw_midline(image, x1,x2,x3,x4, y1, y2, y3, y4)
	return angle

#identify lines
def hough_transformation(image):
	img = image.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray,50,150,apertureSize = 3)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

	lines = cv2.HoughLinesP(gray,7,np.pi/180,100,200,10)
	for x1, y1, x2, y2 in lines[0]:
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255),5)
	#angle_between(x1,x2,y1,y2,image)
	for x3, y3, x4, y4 in lines[1]:
		cv2.line(img, (x3,y3), (x4,y4), (0,0,255),5)
	draw_midline(image, x1, x2, x3, x4, y1, y2, y3, y4)
	img = cv2.imwrite('houghlines.jpg',img)
	img = cv2.imread('houghlines.jpg', 0)
	return img


#get image
def get_image(image):
	img = cv2.imread(image,0)
	return img

#save image
def save_image(image):
	cv2.imwrite("editedImage.jpg", image)
	img = cv2.imread("editedImage.jpg")
	return img 	
	
#height and width
def height_and_width(image):
	height, width = image.shape[:2]
	return (height, width)

#convert color image to binary black and white
def convert_to_binary(image):
	#img = cv2.imread(image, 0)
	retval, img = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
	return img

#crop image
def crop_image(image):
	height, width = image.shape[:2]
	x1 = int(width-(3*width/4))
	x2 = int(width-(width/4))
	y1 = int(height/2)
	y2 = int(height-(height/5))
	roi = image[y1:y2, x1:x2]
	return roi	

#resize image
def resize_image(image):
	r = 500.0/image.shape[1]
	dim = (500, int(image.shape[0]*r))
	resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	return resized

#draw line between hough lines
def draw_midline(image, x1, x2, x3, x4, y1, y2, y3, y4):
	startX = int((x1+x4)/2)
	print(startX)
	endX = int((x2+x3)/2)
	startY = y1
	endY = y2
	
	slope = (startY-endY)/(startX-endX)
	angle = math.atan(slope)
	angle = math.degrees(angle)
	print("angle is")
	print(angle)
	img = cv2.line(image, (startX, startY), (endX, endY), (255,0,0),5)
	cv2.imwrite("final.jpg", img)
	return img
	
def main():
	img = get_image(sys.argv[1])
	img = resize_image(img)
	img = save_image(img)
	img = cv2.imread("editedImage.jpg")
	img = hough_transformation(img)
	img = cv2.imread("houghlines.jpg")
	cv2.imshow("image", img)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()


