
# Quick demo that shows how to take photographs with the RoboMaster TT drones.
#   Notice that we are communicating with the drone directly. Of course, you'll
#   want to create a capability in your drone controller class.

from djitellopy import Tello
import threading
import cv2
import time



def process_video_feed(drone, stop_thread_event, display_video_live=False):

    movie_name = 'drone_capture.avi'
    movie_codec = cv2.VideoWriter_fourcc(*'mp4v')
    movie_fps = 20
    frame_wait = 1 / movie_fps
    movie_size = (360, 240)

    print("Thread started")
    drone.streamon()
    camera = drone.get_frame_read()
    movie = cv2.VideoWriter(movie_name, movie_codec, movie_fps, movie_size, True)
    time_prev = time.time()
    if display_video_live:
        cv2.namedWindow("Drone Video Feed")
    print("Video feed started")
    
    while not stop_thread_event.isSet():

        time_curr = time.time()
        time_elapsed = time_curr - time_prev
        if time_elapsed > frame_wait:
            image = camera.frame
            image = cv2.resize(image, movie_size)
            if display_video_live:
                cv2.imshow("Drone Video Feed", image)
            cv2.waitKey(1)
            movie.write(image)
            time_prev = time_curr

        if display_video_live:
            cv2.waitKey(5)
        else:
            time.sleep(0.005)

    print("Stopping video feed")
    drone.streamoff()
    movie.release()
    print("Thread finished")


# Connect to the drone and turn on the motors to cool down the drone
drone = Tello()
drone.connect()
print(f"Battery level is {drone.get_battery()}%")

# Start the video recording
stop_video_event = threading.Event()
video_thread = threading.Thread(target=process_video_feed, args=(drone, stop_video_event, True))
video_thread.setDaemon(True)
video_thread.start()

# Fly a simple mission
drone.takeoff()
drone.rotate_clockwise(90)
drone.rotate_counter_clockwise(180)
drone.rotate_clockwise(90)
drone.move_forward(30)
drone.move_back(30)
drone.land()

# Stop the video recording and wait for it to finsih
stop_video_event.set()
video_thread.join(0.5)

print("Destroying all picture windows")
cv2.destroyAllWindows()

# Turn off the video stream and the drone

print("Closing connection")
print(f"Battery level is {drone.get_battery()}%")
drone.end()
print("Mission Complete")