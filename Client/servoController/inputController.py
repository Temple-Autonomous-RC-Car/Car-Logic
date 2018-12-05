from time import sleep

from board import SCL, SDA
import busio
import sys

# Import the PCA9685 module. Available in the bundle and here:
#   https://github.com/adafruit/Adafruit_CircuitPython_PCA9685
from adafruit_pca9685 import PCA9685
from adafruit_motor import servo

STEERING_MAX_POS = 60
STEERING_ZERO = 50
STEERING_MAX_NEG = 40

THROTTLE_SCALE = 0.5

i2c = busio.I2C(SCL, SDA)
pca = PCA9685(i2c)
pca.frequency = 60

motor = servo.ContinuousServo(pca.channels[0])
steering = servo = servo.Servo(pca.channels[2], actuation_range=90)
    
def stop():
    motor.throttle = 0

def drive(amount):
    motor.throttle = amount * THROTTLE_SCALE

def steer(amount):
    steer = STEERING_ZERO + amount * 10
    servo.angle = steer
def strongStop():
    motor.throttle = 0
    exit()
def main():
    print("Args ", sys.argv)
    if(len(sys.argv) != 3):
        print("Usage: <drive|steer> <throttle amount| steering amount>")
    elif(sys.argv[1] == 'drive'):
        driveAmt = float(sys.argv[2])
        if(driveAmt >= -1 and driveAmt <= 1):
            drive(driveAmt)
            sleep(1)
            stop()
        else:
            print("Usage: <drive|steer> <throttle amount| steering amount>")
    elif(sys.argv[1] == 'steer'):
        steerAmt = float(sys.argv[2])
        if(steerAmt >= -1 and steerAmt <= 1):
            steer(steerAmt)
            sleep(2)
            steer(0)
        else:
            print("Usage: <drive|steer> <throttle amount| steering amount>")
    else:
        print("Usage: <drive|steer> <throttle amount| steering amount>")
if __name__ == "__main__":
    main()

