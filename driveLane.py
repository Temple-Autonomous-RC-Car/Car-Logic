import edgeDetect.live_image as live_image
from servoController.inputController import drive, steer, stop
from time import sleep
steer(0)
drive(0)


leftSteer = -.80
rightSteer = .80
centerSteer = 0

live_image.run()
try:
    while(True):
        #drive(.23)
        angle = live_image.liveAngle
        print("Got angle of %d" % (angle))
        if(angle > -87 and angle < 0):
            steer(leftSteer)
            sleep(.5)
        elif(angle < 87 and angle >0):
            steer(rightSteer)
            sleep(.5)
        else:
            steer(0)
            sleep(.5)
except KeyboardInterrupt:
    steer(0)
    drive(0)
    exit()
