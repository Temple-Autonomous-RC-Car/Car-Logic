#modified to show example of server communication
#!/usr/bin/python
"""
Released under the MIT License
Copyright 2015 MrTijn/Tijndagamer
Modified by Aaron Geller
"""

# Import the MPU6050 class from the MPU6050.py file
from MPU6050 import MPU6050
#from time import sleep
import time
import socket
import sys

#needed for server communication
import sendToServer

# Create a new instance of the MPU6050 class
sensor = MPU6050(0x68)
highestOutX = 0
highestOutY = 0
highestOutZ = 0

try:

    while True:
        time.sleep(3)
        accel_data = sensor.get_accel_data()
        gyro_data = sensor.get_gyro_data()
        temp = sensor.get_temp()

        print("Accelerometer data")
        #print("x: " + str(accel_data['x']))
        #print("y: " + str(accel_data['y']))
        #print("z: " + str(accel_data['z']))
        if(abs(accel_data['x']) > abs(highestOutX)):
            highestOutX = accel_data['x']
        if(abs(accel_data['y']) > abs(highestOutY)):
            highestOutY = accel_data['y']
        if(abs(accel_data['z']) > abs(highestOutZ)):
            highestOutZ = accel_data['z']
        # print("Gyroscope data")
        # print("x: " + str(gyro_data['x']))
        # print("y: " + str(gyro_data['y']))
        # print("z: " + str(gyro_data['z']))
        xout = "x: "+str(highestOutX) 
        print(xout)

        #send data to server as bitwise encoded string
        sendToServer.s.send(xout.encode())

        yout = "y: "+ str(highestOutY) 
        print(yout)

        #ensure proper string formatting on server side
        yout = yout + "\n"
        #send data to server
        sendToServer.s.send(yout.encode())

        #ensure proper formatting on server side
        zout = "z: "+ str(highestOutZ) + "\n"
        print(zout)
        #send data to server
        sendToServer.s.send(zout.encode())
        #print("Temp: " + str(temp) + " C")

except KeyboardInterrupt:
    sendToServer.s.close()
    sys.exit()
