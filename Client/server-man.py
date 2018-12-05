import socket
import _thread
import sys
import os
import time
import struct
import select
import subprocess
import servoController.inputController as inControl

inControl.steer(0)
def empty_socket(sock):
    """remove the data present on the socket"""
    input = [sock]
    while 1:
        inputready, o, e = select.select(input,[],[], 0.0)
        if len(inputready)==0: break
        for s in inputready: s.recv(1)


#loop to mannage connected socket input
def on_new_client(clientsocket,addr):
    while True:
        try:
            size = struct.unpack("i", clientsocket.recv(struct.calcsize("i")))[0]
            data = ""
            #cmd = msg.decode(encoding='utf-8')
            while len(data) < size:
                msg = clientsocket.recv(size - len(data))
                empty_socket(clientsocket)
                if not msg:
                    break
                data += msg.decode()
            print(data)
            inControl.drive(.26)
        except:
            inControl.stop()
            #clientsocket.close()
            break

        #subprocess.call(data, shell=True)
        if "steer" in data:
           words = data.split()
           amt = float(words[1])
           inControl.steer(amt)


        #time.sleep(0.1)
    clientsocket.close()

#created and bind socket
s = socket.socket()
port = 12346
s.bind(('',port))
#visual feedback that the server is running
print("Server Listening")

try:
    #listen for an allow 5 connections
    s.listen(5)
    while True:
        c,addr = s.accept()
        print ("connect from ",addr)

        #pass newly connected socket to its own thread
        _thread.start_new_thread(on_new_client,(c,addr))

    s.close()

except KeyboardInterrupt:
    c.close()
    s.close()
    sys.exit()
