from servoController.inputController import stop,drive,steer
from sensors.proximitySensor import *
from time import sleep

arr = [200,200,200,200,200,200,200,200,200,200] #200cm 5 value scrolling window for fiter
def medianFilt(val):
    for i in range(0,9):
        arr[i] = arr[i+1]
    arr[9] = val
    temp = arr.copy()
    temp.sort()
    print("Array is " + str(arr))
    return temp[5]

getDistance()
getDistance()
steer(-.22)
drive(0.38)
while(True):
    val = getDistance()
    val = medianFilt(val)
    print("Distance is: %.2f" % val)
    if(val < 85.0):
        drive(-1)
        #sleep(1)
        #stop()
        break
        print("Stopped")
print("Done")
cleanUp()    
exit()

