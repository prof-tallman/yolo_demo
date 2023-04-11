
import os
import sys
import time
import platform

from dronekit import connect, VehicleMode, APIException
from pymavlink import mavutil

class Pihawk:

    def __init__(self, connection_str='/dev/ttyAMA0', baud=115200):

        self.drone = None
        try:
            print(f"Attempting to connect to drone at '{connection_str}'")
            self.drone = connect(connection_str, wait_ready=True, heartbeat_timeout=2)
            print(f"Battery is at {self.drone.battery}")
        except:# APIException as e:
            #print(f"Error: {str(e)}")
            raise RuntimeError('Cannot connect to drone')

    

    def __del__(self):

        if self.drone:

            if self.drone.mode != 'LAND':
                self.land()

            if not self.drone.armed:
                self.disarm()

            self.drone.close()


    def arm(self):

        print("Attempting to arm drone")
        count = 0
        while not self.drone.is_armable and count < 15:
            print(f"Waiting for vehicle to initialize... {count}s")
            time.sleep(1)
            count += 1
        if not not self.drone.is_armable:
            return self.drone.armed
        print("Vehicle is now armable")

        self.drone.mode = VehicleMode("GUIDED")  
        count = 0
        while self.drone.mode != 'GUIDED':
            print(f"Waiting for drone to enter GUIDED flight mode... {count}s")
            time.sleep(1)
            count += 1
        print("Vehicle now in GUIDED MODE")

        self.drone.armed = True
        count = 0
        while not self.drone.armed and count < 5:
            print(f"Waiting for vehicle to arm itself... {count}s")
            time.sleep(1)
            count += 1

        print(f"Drone is now ready")
        return self.drone.armed


    def disarm(self):
        print("Attempting to disarm drone")
        if self.drone == None:
            raise 
        self.drone.armed = False
        count = 0
        while self.drone.armed and count < 10:
            print(f"Waiting for vehicle to disarm itself... {count}s")
            time.sleep(1)
            count += 1
        return self.drone.armed
    

    def takeoff(self, takeoff_height=1):
        print("Attempting to takeoff drone")
        height = 0
        count = 0
        if height > 10:
            print(f"Capping takeoff height from {takeoff_height}m to 10m")
            takeoff_height = 10

        self.drone.simple_takeoff(takeoff_height)
        timeout = takeoff_height * 5
        while height < 0.95 * takeoff_height and count < timeout:
            height = self.drone.location.global_relative_frame.alt
            print(f"Drone is now at height {height}... {count}s")
            time.sleep(1)
            count += 1
        return height


    def land(self):
        print("Attempting to land drone")
        self.drone.mode = VehicleMode("LAND")
        count = 1
        while self.drone.mode != 'LAND' and count < 30:
            print(f"Waiting for drone to land... {count}s")
            time.sleep(1)
            count += 1
        return self.drone.mode


marty = Pihawk(connection_str='udp:127.0.0.1:14550')
marty.arm()
marty.takeoff(1)
print("Takeoff complete, showing off for 5s")
time.sleep(5)
print("Done showing off, now landing")
marty.land()
marty.disarm()
