from PIL import Image
import matplotlib.pyplot as plt
import cv2
import sys
import numpy as np

#identify lines
def hough_transformation(image):
	img = image.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray,50,150,apertureSize = 3)
	#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#edges = cv2.Canny(gray, 50, 150, apertureSize = 3)

	lines = cv2.HoughLinesP(gray,1,np.pi/180,100,200,10)
	for x1, y1, x2, y2 in lines[0]:
		"""a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))"""
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255),5)
	print(x1,x2,y1,y2)
	for x1, y1, x2, y2 in lines[1]:
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255),5)
	img = cv2.imwrite('houghlines2.jpg',img)
	img = cv2.imread('houghlines2.jpg', 0)
	print(x1,x2,y1,y2)
	"""for rho,theta in lines[1]:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))
		cv2.line(img, (x1,y1), (x2,y2), (0,0,255),5)

	print(theta)
	img = cv2.imwrite('houghlines.jpg',img)
	img = cv2.imread('houghlines.jpg', 0)
"""
	return img


#get image
def get_image(image):
	img = cv2.imread(image,0)
	return img

#reduce noise in an image
def reduce_noise(image):
	img = cv2.imread(image)
	img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
	cv2.imwrite("editedImage.jpg", img)
	return img

#canny edge detection
def canny_edge_detection(image):
	edges = cv2.Canny(image, 100, 200)
	plt.subplot(121), plt.imshow(image, cmap = 'gray')
	plt.subplot(122), plt.imshow(edges, cmap = 'gray')
	return edges

#undistort image

#save image
def save_image(image):
	cv2.imwrite("editedImage.jpg", image)
	img = cv2.imread("editedImage.jpg")
	return img 	
	
#height and width
def height_and_width(image):
	height, width = image.shape[:2]
	return (height, width)

#draw center line through image
def draw_center_line(image):
	height, width = image.shape[:2]
	startX =int(width/2)
	startY = 0
	endX = startX
	endY = height
	img = cv2.line(image, (startX, startY), (endX, endY), (255,0,0),25)
	cv2.imwrite("final.jpg", img)
	return img	

#convert color image to binary black and white
def convert_to_binary(image):
	#img = cv2.imread(image, 0)
	retval, img = cv2.threshold(image, 140, 255, cv2.THRESH_BINARY)
	return img

#convert binary image to binary array
def convert_to_array(image):
	np_img = np.array(image)
	np_img[np_img > 0] = 1
	return np_img

#crop image
def crop_image(image):
	height, width = image.shape[:2]
	x1 = int(width/4)
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

#finding the edges from an image that has been turned into a binary image

#gaussian images
def convert_to_gaussian(image):
	img = cv2.imread(image, 0)
	img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
	return img

def main():
	img = get_image(sys.argv[1])
	img = resize_image(img)
	#img = crop_image(img)
	img = save_image(img)
	img = cv2.imread("editedImage.jpg")
	#img = hough_transformation(img)
	#img = reduce_noise(sys.argv[1])
	#img = cv2.imread("editedImage.jpg", 0)
	#img = canny_edge_detection(img)
	img = convert_to_binary(img)
	#img = canny_edge_detection(img)
	#img = convert_to_gaussian(sys.argv[1])
	#img = resize_image(img)
	#img = crop_image(img)
	#print(height_and_width(img))
	img = save_image(img)
	img = cv2.imread("editedImage.jpg")
	img = hough_transformation(img)
	img = cv2.imread("houghlines2.jpg")
	img = draw_center_line(img)
	#canny_edge_detection(img)
	cv2.imshow("image", img)
	cv2.waitKey(5000)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()


