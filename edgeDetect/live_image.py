import threading
import edgeDetect.still_image as still_image
import cv2
from picamera import PiCamera
import time
from picamera.array import PiRGBArray

liveAngle = 90

def run():
    threading.Thread(target=frameAnalysisThread).start()

def frameAnalysisThread():    
    camera = PiCamera()
    camera.framerate = 30
    camera.resolution = (720,720)
    rawCapture = PiRGBArray(camera, size=(720,720))

    time.sleep(0.1)

    #cv2.startWindowThread()
    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    for frame in camera.capture_continuous(rawCapture, format="rgb", use_video_port=True):
        image = frame.array
        frame = still_image.process_frame(image)

        #cv2.imshow("Frame", frame)
        global liveAngle 
        liveAngle = still_image.angleVar
        #print("Live angle is %d" % (liveAngle))
        rawCapture.truncate(0)
       # key = cv2.waitKey(1) & 0xFF
        #if(key == ord("q")):
         #   break
    #cv2.destroyAllWindows()
def main():
    run()
if(__name__ == "__main__"):
    main()
