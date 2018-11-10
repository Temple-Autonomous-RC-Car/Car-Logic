import socket
import sys

TCP_IP = '10.42.0.1'
TCP_PORT = 12346

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


try:
    while True:

        line = sys.stdin.readline()
        if line:
            s.send(line.encode())
        else:
            break

except KeyboardInterrupt:
    s.close()
    sys.exit()
    
