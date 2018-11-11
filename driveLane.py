import edgeDetect.live_image as live_image
from servoController.inputController import drive, steer, stop
from time import sleep
steer(0)
drive(0)


c = "CENTER"
l = "LEFT"
r = "RIGHT"

leftSteer = 1
rightSteer = -1
centerSteer = 0

live_image.run()
try:
    drive(.3)
    while(True):
        angle = live_image.liveAngle
        botOfLine = live_image.liveLine
        #print("Got angle of %d" % (angle))
        if(angle > -87 and angle < 0):
            #drive(.26)
            if(botOfLine == c or botOfLine == l): 
                steer(leftSteer)
            elif(botOfLine == r):
                steer(rightSteer)
            #sleep(.2)
        elif(angle < 87 and angle >0):
            #drive(.26)
            if(botOfLine == l): 
                steer(leftSteer)
            elif(botOfLine == c or botOfLine == r):
                steer(rightSteer)
            #sleep(.2)
        else:
            steer(centerSteer)
            #drive(.28)
            #sleep(.2)
except KeyboardInterrupt:
    steer(0)
    drive(-1)
    exit()
