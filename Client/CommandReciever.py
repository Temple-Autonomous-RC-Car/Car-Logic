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
import sensors.proximitySensor as prox
import pigpio
import time
import numpy as np


Stopped = True

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

def obstacleThread(q):
    THRESH = 2.0
    distances = [300,300,300]
    pi = pigpio.pi()
    sonar = prox.ranger(pi, 23, 24)
    t = threading.currentThread()
    localStopped = False
    while getattr(t, "do_run", True):
        try:
            distances[2] = distances[1]
            distances[1] = distances[0]
            distance = (.0343 * sonar.read()) / 2
            distances[0] = distance
            time.sleep(0.03)
            value = (abs(distances[0] - distances[1]) < THRESH) & (abs(distances[2] - distances[1]) < THRESH) & (abs(distances[0] - distances[2]) < THRESH)
            if(value & (np.mean(distances)<50)):
                entry = CommandEntry(str(1),str(time.time()), "drive", str(-1))
                q.put(entry)
                localStopped = True
            elif(localStopped):
                localStopped = False
                entry = CommandEntry(str(1),str(time.time()), "drive", str(.28))
                q.put(entry)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            break

#loop to mannage connected socket input
def on_new_client(clientsocket,addr, q):
    print("Socket recv starting.")
    t = threading.currentThread()
    clientsocket.settimeout(120.0)
    while getattr(t, "do_run", True):
        try:
            dumpFlag = getattr(t, "dump_commands", False)
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
            if(not dumpFlag):
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

def socketAccept(q, worker_list):
    
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
            worker_list.append(newThread)
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

def doCommand(cEntry, q, workerList):
    global Stopped
    HARD_TURN = 1
    WAIT_AMT = 2.1
    #bits = cString.split()
    """
    <priority> <timestamp> <steer|drive|stop> <amt>
    """
    command = cEntry.command
    amt = float(cEntry.amount)
    print("Command is %s amount is %.2f"%(command,amt))
    if(command == "steer"):
        inControl.steer(amt)
    elif(command == "drive"):

        if(amt<=0 and Stopped == True):
            amt = 0
        elif(amt<=0 and Stopped == False):
            Stopped = True
        else:
            Stopped = False
        print("Amount %f" % amt)
        inControl.drive(amt)
        
    elif(command == "stop"):
        inControl.stop()
        time.sleep(amt)
    elif(command =="stopright"):
        Stopped = True
        signalDump(workerList, True)
        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.steer(HARD_TURN * -1)
        inControl.drive(.28)
        time.sleep(WAIT_AMT)
        inControl.steer(0)
        signalDump(workerList, False)
        Stopped = False
        #time.sleep(.3)
    elif(command =="stopleft"):
        Stopped = True
        signalDump(workerList, True)
        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.steer(HARD_TURN * 1)
        inControl.drive(.3)
        time.sleep(WAIT_AMT)
        #inControl.drive(.26)
        inControl.steer(0)
        signalDump(workerList, False)
        inControl.drive(0)
        time.sleep(.6)
        inControl.drive(.28)
        #time.sleep(.3)
        Stopped = False
    elif(command =="stopcenter"):
        Stopped=True
        #ts0=cEntry.timestamp
        signalDump(workerList, True)
        inControl.drive(-1) #TODO
        time.sleep(amt)
        inControl.drive(.28)
        inControl.steer(0)
        #time.sleep(1)
        signalDump(workerList, False)
        #ts1 = time.time()
        Stopped=False
        
    else:
        print("Command not recognized")
def signalDump(threads, boolean):
    for thread in threads:
        thread.dump_commands = boolean
        

def main():
    q = queue.PriorityQueue()
    workerList = []
    receiverThread = threading.Thread(target = socketAccept, args=(q,workerList))
    receiverThread.start()
    obsThread = threading.Thread(target = obstacleThread, args=(q,))
    obsThread.start()
    workerList.append(obsThread)
    while True:
        try:
            value = q.get(True, 0.05)
            doCommand(value,q, workerList)
        except queue.Empty:
            continue
        except KeyboardInterrupt:
            inControl.stop()
            receiverThread.do_run = False
            obsThread.do_run = False
            receiverThread.join()
            exit()
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            inControl.stop()
            receiverThread.do_run = False
            obsThread.do_run = False
            receiverThread.join()
            obsThread.join()
            exit()
if(__name__ == "__main__"):
    main()
