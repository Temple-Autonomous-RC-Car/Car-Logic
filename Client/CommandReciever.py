import socket
import threading
import sys
import os
import queue
import struct
import functools
import traceback
sys.path.append('..')
import servoController.inputController as inControl
import time
@functools.total_ordering
class CommandEntry(object):
    """
    Object that stores command entry information for use in a prio queue.
    If the priority of two commands is equal, the one with the lowest timestamp is taken.
    """
    def __init__(self, priority, timestamp, command, amount):
        self.priority = priority
        self.timestamp = timestamp
        self.command = command
        self.amount = amount
        return
    def __lt__(self, other):
        if(self.priority < getattr(other, 'priority', other)):
            return True
        elif(self.priority == getattr(other, 'priority', other)):
            return self.timestamp < getattr(other, 'timestamp', other)
        else:
            return False
    def __eq__(self, other):
        if(self.priority == getattr(other, 'priority', other)):
            if(self.timestamp == getattr(other, 'timestamp', other)):
                return True
        return False



#loop to mannage connected socket input
def on_new_client(clientsocket,addr, q):
    print("Socket recv starting.")
    t = threading.currentThread()
    clientsocket.settimeout(20.0)
    while getattr(t, "do_run", True):
        try:
            inc = clientsocket.recv(struct.calcsize("i"))
            if len(inc) == 0:
                print("Socket closed")
                clientsocket.close()
                break
            size = struct.unpack("i", inc)[0]
            data = ""
            #cmd = msg.decode(encoding='utf-8')
            while len(data) < size:
                msg = clientsocket.recv(size - len(data))
                #empty_socket(clientsocket)
                if not msg:
                    break
                data += msg.decode()
            if not data:
                continue
            '''
            Data should be in form:
            <priority> <drive|steer|stop> <amt>
            Drive will set the car to drive at a constant amount 
            Steer will turn the car left and right 
            Stop will make the car sleep for 3 seconds.
            '''
            words = data.split()
            q.put(CommandEntry(words[0],words[1],words[2],words[3]))
        except socket.timeout:
            print("Socket closed")
            clientsocket.close()
            break
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            #inControl.stop()
            continue
    print("Socket recv closing.")
    clientsocket.close()

def socketAccept(q):
    
    #listen for an allow 5 connections
    #created and bind socket
    s = socket.socket()
    port = 12346
    s.bind(('',port))
    #visual feedback that the server is running
    print("Server Listening")
    s.listen(5)
    s.settimeout(.5)
    t = threading.currentThread()
    threads = []
    while getattr(t, "do_run", True):
        try:
            c,addr = s.accept()
            print ("connect from ",addr)
            #pass newly connected socket to its own thread
            newThread = threading.Thread(target=on_new_client,args=(c,addr, q))
            threads.append(newThread)
            newThread.start()
        except socket.timeout:
            continue
        except:
            break
    print("Socket accept closing.")
    c.close()
    s.close()
    for thread in threads:
        if thread.is_alive():
            thread.do_run = False
            thread.join()

def doCommand(cEntry, q):
    HARD_TURN = 1
    WAIT_AMT = 1.8
    #bits = cString.split()
    """
    <priority> <timestamp> <steer|drive|stop> <amt>
    """
    command = cEntry.command
    amt = float(cEntry.amount)
    if(command == "steer"):
        inControl.steer(amt)
    elif(command == "drive"):
        inControl.drive(amt)
    elif(command == "stop"):
        inControl.stop()
        time.sleep(amt)
    elif(command =="stopright"):
        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.steer(HARD_TURN * -1)
        inControl.drive(.28)
        time.sleep(WAIT_AMT)
        inControl.steer(0)
        time.sleep(.3)
    elif(command =="stopleft"):
        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.steer(HARD_TURN * 1)
        inControl.drive(.28)
        time.sleep(WAIT_AMT)
        inControl.steer(0)
        time.sleep(.3)

    elif(command =="stopcenter"):
        #ts0=cEntry.timestamp

        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.drive(.28)
        inControl.steer(0)
        time.sleep(1)

        #ts1 = time.time()
        
    else:
        print("Command not recognized")
    
    
def main():
    q = queue.PriorityQueue()
    receiverThread = threading.Thread(target = socketAccept, args=(q,))
    receiverThread.start()
    while True:
        try:
            value = q.get(True, 0.05)
            doCommand(value,q)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            inControl.stop()
            receiverThread.do_run = False
            receiverThread.join()
            exit()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            inControl.stop()
            receiverThread.do_run = False
            receiverThread.join()
            exit()
if(__name__ == "__main__"):
    main()
