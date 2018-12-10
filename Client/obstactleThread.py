import sys
sys.path.append("..")
import servoController.inputController as inControl
import sensors.proximitySensor as prox
import pigpio
import time
import math
import numpy as np

THRESH = 2.0
#stopped = True

def main():
    distances = [300,300,300]
    global stopped
    pi = pigpio.pi()
    sonar = prox.ranger(pi, 23, 24)
    while True:
        distances[2] = distances[1]
        distances[1] = distances[0]
        distance = (.0343 * sonar.read()) /2
        distances[0] = distance
        print(distances)
        time.sleep(0.03)
        value = (abs(distances[0] - distances[1]) < THRESH) & (abs(distances[2] - distances[1]) < THRESH) & (abs(distances[0]- distances[2]) < THRESH) 
        if(value & (np.mean(distances) < 50)):
            stopCar()
        else:
            stopped = False
            #inControl.drive(.26)
        
    sonar.cancel()
    pi.stop()


def stopCar():
    global stopped
    #print("Stopped")
    if(stopped == False):
        inControl.drive(-1)
        time.sleep(0.3)
        inControl.drive(0)
        stopped = True
    else:
        inControl.drive(0)

if(__name__ =="__main__"):
    inControl.drive(0)
    main()
