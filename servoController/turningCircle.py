from inputController import drive, steer, stop
from time import sleep

try:
    steer(1)
    drive(.28)
    while(True):
        i= 0

except KeyboardInterrupt:
    stop()
    exit()
