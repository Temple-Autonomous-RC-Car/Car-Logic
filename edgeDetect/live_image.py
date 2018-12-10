import threading
import edgeDetect.still_image as still_image
import cv2
from picamera import PiCamera
import time
from picamera.array import PiRGBArray

liveAngle = 90
liveLine = "CENTER"
def run():
    threading.Thread(target=frameAnalysisThread, daemon = True).start()

def frameAnalysisThread():    
    camera = PiCamera()
    camera.framerate = 60
    camera.resolution = (900,620)
    camera.brightness = 60
    rawCapture = PiRGBArray(camera, size=(900,620))

    time.sleep(0.1)

    #cv2.startWindowThread()
    #cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        x = int(image.shape[1])
        y = int(image.shape[0])
        y_top = int(y/3)
        #cropped = image[y_top:y, 0:x]
        #crop_img = cv2.imread(cropped)  
        frame = still_image.process_frame(image)

       # cv2.imshow("Frame", frame)
        global liveAngle 
        liveAngle = still_image.angleVar
        print("Live angle is %d" % (liveAngle))
        global liveLine
        liveLine = still_image.bottomOfLine
        print("Bottom of line is in %s" % (liveLine))
        rawCapture.truncate(0)
        #key = cv2.waitKey(1) & 0xFF
        #if(key == ord("q")):
        #    break
   # cv2.destroyAllWindows()
def main():
    run()
if(__name__ == "__main__"):
    main()
