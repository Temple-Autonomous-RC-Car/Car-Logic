import cv2
import numpy as np
import matplotlib.pyplot as plt


image = cv2.imread("img2.jpg")
cv2.circle(image, (1100,700), 5, (0,0,255), -1)
cv2.circle(image, (1600, 700), 5, (0,0,255), -1)
cv2.circle(image, (400, 1450), 5, (0,0,255), -1)
cv2.circle(image, (2340, 1450), 5, (0,0,255), -1)
pts1 = np.float32([[1100,700],[1600,700],[400,1450],[2340,1450]])
pts2 = np.float32([[0,0],[1500,0],[0,2400],[1500,2400]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)

result = cv2.warpPerspective(image, matrix, (1500,2400))
cv2.imshow("result", result)

height, width = image.shape[:2]
print(height)
print(width)

cv2.waitKey(5000)
