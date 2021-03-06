#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML



angleVar = 90
bottomOfLine = "CENTER"
_noLines = 0
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).
    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.
    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    ret = weighted_img(line_img, img)        
    return ret
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([])) #minLineLength=min_line_len, maxLineGap=max_line_gap)
    #if(lines.any() == None):
    #    print("No lines found.")
    #    exit()
    #draw_lines(line_img, lines)
    #return line_img
    if lines is None:
        print("No lines found")
    return lines

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    `initial_img` should be the image before any processing.
    The result image is computed as follows:
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def draw_avg_lines(lines, image):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            #print("(%d , %d) and (%d , %d)" % (x1,y1,x2,y2))
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            if (x1 == x2):
                slope = 0
            else:    
                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
            if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                continue
            if slope < 0: 
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else: # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    min_y = image.shape[0] * (3/8) #<-- Just below the horizon
    max_y = image.shape[0] # <-- The bottom of the image
    areLines = True
    
    if not left_line_x:
        print("no left line!")
        areLines = False
    else:
        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))
    if not right_line_x:
        print("no right line!")
        areLines=False
    else:
        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))
    if(areLines == False):
        global bottomOfLine
        bottomOfLine = "NONE"
        return image
    drawn_img = draw_lines(image,[[ [left_x_start, max_y, left_x_end, min_y],[right_x_start, max_y, right_x_end, min_y] ]], thickness = 20)
    
    #cv2.namedWindow("processedImage",cv2.WINDOW_NORMAL)
    #processed = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2RGB)
    #cv2.imshow("processedImage",processed)

    rightLine = [right_x_start, max_y, right_x_end, min_y]
    leftLine = [left_x_start, max_y, left_x_end, min_y]
    endImage=find_midline_angle(drawn_img, rightLine, leftLine)

    
    return endImage

def get_bottom_point(x_length, x_pos):
    global bottomOfLine
    if(x_pos < x_length/2 -10):
        bottomOfLine = "LEFT"
    elif(x_pos > x_length/2 + 10):
        bottomOfLine = "RIGHT"
    else:
        bottomOfLine = "CENTER"
    

def find_midline_angle(image, line1, line2):
    midLine = (np.array(line1) + np.array(line2)) /2
    angles =[]
    x1, y1, x2, y2 =  midLine
    #print("Points are: (%d,%d) and (%d,%d)" % (x1,y1,x2,y2))
    #for x1,y1,x2,y2 in lines:
    if(x1 < x2):
        slope = (y1-y2)/(x1-x2)
    else:
        slope = (y2-y1)/(x2-x1)
    angle = math.atan(slope)
    angle = math.degrees(angle)
    #get Bottom area of line
    if(y1 > y2):
        get_bottom_point(image.shape[1], x1)
    else:
        get_bottom_point(image.shape[1], x2)
    img = cv2.line(image, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0),20)
    #cv2.namedWindow("Midline", cv2.WINDOW_NORMAL)
    #cv2.imshow("Midline",img)
    
    #print("Slope is %f" % (slope))
    #print("Angle is %d" % (angle))
    global angleVar
    angleVar = angle
    return img

def process_frame(image):
    global first_frame

    gray_image = grayscale(image)
   # cv2.startWindowThread()
    #cv2.namedWindow("grayScale",cv2.WINDOW_NORMAL)
    #cv2.imshow("grayScale",gray_image)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #hsv = [hue, saturation, value]
    #more accurate range for yellow since it is not strictly black, white, r, g, or b

    lower_blue = np.array([90,90,90])
    upper_blue = np.array([110,255,255])

    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    #mask_white = cv2.inRange(gray_image, 200, 255)
    #mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray_image, mask_blue)
   # cv2.namedWindow("maskedImage",cv2.WINDOW_NORMAL)
   # cv2.imshow("maskedImage",mask_yw_image)
    kernel_size = 1
    gauss_gray = gaussian_blur(mask_yw_image,kernel_size)

    #same as quiz values
    low_threshold = 50
    high_threshold = 150
    canny_edges = canny(gauss_gray,low_threshold,high_threshold)
   # cv2.namedWindow("cannyImage",cv2.WINDOW_NORMAL)
   # cv2.imshow("cannyImage",canny_edges)
    imshape = image.shape
    lower_left = [0,imshape[0]]
    lower_right = [imshape[1], imshape[0]]
    top_left = [imshape[1]/7,imshape[0]/4]
    top_right = [imshape[1]-imshape[1]/7, imshape[0]/4]
    vertices = [np.array([lower_left,top_left,top_right,lower_right],dtype=np.int32)]
    roi_image = region_of_interest(canny_edges, vertices)
    #cv2.namedWindow("roi", cv2.WINDOW_NORMAL)
    #cv2.imshow("roi", roi_image)
    #rho and theta are the distance and angular resolution of the grid in Hough space
    #same values as quiz
    rho = 2
    theta = np.pi/180
    #threshold is minimum number of intersections in a grid for candidate line to go to output
    threshold = 40
    min_line_len =5
    max_line_gap = 2

    lines = hough_lines(canny_edges, rho, theta, threshold, min_line_len, max_line_gap)
    if(lines is None):
        return image
    #esult = draw_lines(image, lines, [0,0,255], thickness = 5)
    result = draw_avg_lines(lines, image)
    #rightLine, leftLine = avgLines
    #find_midline_angle(image,rightLine, leftLine)
    
    
    #cv2.namedWindow("houghs", cv2.WINDOW_NORMAL)
    #cv2.imshow("houghs",draw_lines(image, lines))
    return result

def main():
    for source_img in os.listdir("test_images/"):
        print(source_img)
        image = mpimg.imread("test_images/"+source_img)
        image = process_frame(image)
        cv2.namedWindow("Dog", cv2.WINDOW_NORMAL)
        cv2.imshow("Dog", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if(__name__ == "__main__"):
    main()

