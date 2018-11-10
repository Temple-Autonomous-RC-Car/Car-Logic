import socket
import _thread
import sys

#loop to mannage connected socket input
def on_new_client(clientsocket,addr):
    while True:
        msg=clientsocket.recv(1024)
        print(msg.decode(encoding='utf-8'))
    clientsocket.close()

#created and bind socket
s = socket.socket()
port = 12346
s.bind(('',port))

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
