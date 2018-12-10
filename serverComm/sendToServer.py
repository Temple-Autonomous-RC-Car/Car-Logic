import socket
import sys

#IP address of the socket server goes here
TCP_IP = '192.168.137.156'
#Poet reserved for data
TCP_PORT = 12346

#This is the variable that will open the socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))

#This file just needs to be imported in full into the python
#file that will be sending the data.  To send data to the
#server from the Pi just import this file and reference the
# 's' variable to access the socket.
#example: sendToServer.s.send(str(highestOutX).encode())
#You need to reference the imported file then the variable name
#the string being sent needs to be in bitwise format for
#python3 so just add .encode() after a variable holding a
#string or a lowercase b in front of a raw string in
#single quotes example: b'string here'
