import servoController.inputController as inControl
import sys

def main():
    if(len(sys.argv) < 2):
        print("MISSING ARG")
        exit()
    steerVal = float(sys.argv[1])
    inControl.steer(steerVal)

if(__name__ == "__main__"):
    main()
