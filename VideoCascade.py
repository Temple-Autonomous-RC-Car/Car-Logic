#!/usr/bin/env python
# coding: utf-8

# In[7]:


import cv2
import time
from servoController.inputController import stop,drive,steer

scale_factor = 1.3
min_neighbors = 10
min_size = (75, 75)
webcam=True #if working with video file then make it 'False'
 
def detect(path):
    cascade = cv2.CascadeClassifier(path)
    if webcam:
        video_cap = cv2.VideoCapture("http://localhost:8081") # use 0,1,2..depanding on your webcam
    else:
        video_cap = cv2.VideoCapture("stopvideo.MOV")
    
    steer(0)
   # WINDOW_NAME = "Object Detection"
   # cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
   # cv2.startWindowThread()
    drive(0.26)
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()
        
        if (ret==False):
            break
 
        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
        rects = cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors,
                                         minSize=min_size)
        # if at least 1 face detected
        if len(rects) > 0:
            # Draw a rectangle around the faces
            #for (x, y, w, h) in rects:
              #  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
           # print("Width"+str(w))
           # print("Height"+str(h))
           # print()
            drive(-1)
            time.sleep(3)
            print("Object Detecting")
            turn()
            break
            #steer(-1)
            #drive(0.28)
            #time.sleep(1)
            #drive(0)
            # Display the resulting frame
           # r = 1000.0 / img.shape[1]
           # dim = (1000, int(img.shape[0] * r))
           # resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
           # cv2.imshow(WINDOW_NAME, resized)
            #wait for 'c' to close the application
           # if cv2.waitKey(1) & 0xFF == ord('q'):
            #    stop()
             #   break
    
    video_cap.release()
    
def main():
    cascadeFilePath="stop.xml"
    try:
        detect(cascadeFilePath)
    except KeyboardInterrupt: 
        stop()
            
    #cv2.destroyAllWindows()
    cv2.waitKey(1)
 
def turn():
    steer(-1)
    drive(0.28)
    time.sleep(3)
    steer(0)
    drive(0.28)
    time.sleep(1)
    drive(0)

if __name__ == "__main__":
    main()


# In[ ]:




