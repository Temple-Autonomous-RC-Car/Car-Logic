import edgeDetect.live_image as live_image
from servoController.inputController import drive, steer, stop
from time import sleep
steer(0)
drive(0)


leftSteer = -.60
rightSteer = .60
centerSteer = .12

live_image.run()
try:
    while(True):
        drive(.28)
        angle = live_image.liveAngle
        print("Got angle of %d" % (angle))
        if(angle > -88 and angle < 0):
            steer(leftSteer)
            sleep(.2)
        elif(angle < 88 and angle >0):
            steer(rightSteer)
            sleep(.2)
        else:
            steer(0)
            sleep(.2)
except KeyboardInterrupt:
    steer(0)
    drive(0)
    exit()
