#!/usr/bin/python
"""
Released under the MIT License
Copyright 2015 MrTijn/Tijndagamer
Modified by Aaron Geller
"""

# Import the MPU6050 class from the MPU6050.py file
from MPU6050 import MPU6050
from time import sleep

# Create a new instance of the MPU6050 class
sensor = MPU6050(0x68)
highestOutX = 0
highestOutY = 0
highestOutZ = 0
while True:
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
    print("x: "+ str(highestOutX))
    print("y: "+ str(highestOutY))
    print("z: "+ str(highestOutZ))
   #print("Temp: " + str(temp) + " C")
    
