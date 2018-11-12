import edgeDetect.live_image as live_image
from servoController.inputController import drive, steer, stop
from time import sleep
steer(0)
drive(0)


c = "CENTER"
l = "LEFT"
r = "RIGHT"

steerMax = .8
centerSteer = 0

live_image.run()
try:
    drive(.28)
    while(True):
        angle = live_image.liveAngle
        botOfLine = live_image.liveLine
       # if(botOfLine == "NONE"):
       #     drive(-1)
       #     steer(0)
       #     sleep(1)
       #     drive(0)
       #     exit()
        #print("Got angle of %d" % (angle))
        if(angle > -88 and angle < 0):
            #drive(.28)
            if angle > -60.0:
                angle = -60.0
            percent = -60.0 /angle
            steerVal= percent * steerMax
            #drive(.26)
            if(botOfLine == r): 
                steer(-steerVal)
            elif(botOfLine == l):
                steer(steerVal)
            elif(botOfLine == c):
                steer(steerVal/2)
            #sleep(.2)
        elif(angle < 88 and angle >0):
            #drive(.28)
            if(angle < 60.0):
                angle = 60.0
            percent = 60.0/angle
            steerVal= percent * steerMax
            if(botOfLine == r): 
                steer(-steerVal)
            elif(botOfLine == l):
                steer(steerVal)
            elif(botOfLine == c):
                steer(steerVal/2)
            #sleep(.2)
        else:
            steer(centerSteer)
            #drive(.28)
            #sleep(.2)
except KeyboardInterrupt:
    steer(0)
    drive(-1)
    sleep(1)
    drive(0)
    exit()
