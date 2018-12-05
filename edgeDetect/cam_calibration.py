import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1,2)

objpoints = []
imgpoints = []

images = glob.glob("calibration_images/*.jpg")

for fname in images:
	img = cv2.imread(fname)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	ret, corners = cv2.findChessboardCorners(gray, (7,6), None)

	if ret == True:
		objpoints.append(objp)
		
		cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
		imgpoints.append(corners)

		cv2.drawChessboardCorners(img, (7,6), corners, ret)
		cv2.imshow('img', img)
		cv2.waitKey(500)

	cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

image = cv2.imread("calibration_images/left12.jpg")
h, w, = image.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
dst = cv2.remap(image, mapx,mapy, cv2.INTER_LINEAR)

print(roi)
x,y,w,h = roi
print(x)
print(y)
print(w)
print(h)
dst = dst[y:y+h, x:x+w]
cv2.imwrite("calibrate.png", dst)
