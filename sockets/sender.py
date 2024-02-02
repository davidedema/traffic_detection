# Sender Code
import cv2
import socket
import pickle
import struct
import os
import time

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

def calculateFps(last_frame_time, frame) -> int:
    """
    Calculate the fps and put it on the frame

    Args:
        prev_frame_time (float): the previous frame time
        frame (np.ndarray): the frame to put the fps on

    Returns:
        last_frame_time (float): the last frame time
    """

    new_frame_time = time.time()
    fps = 1/(new_frame_time-last_frame_time) 
    last_frame_time = new_frame_time 

    fps = int(fps) 
    fps = str(fps) 

    # putting the FPS count on the frame 
    cv2.putText(frame, fps, (7, 70),cv2.FONT_HERSHEY_SIMPLEX,2,(0, 0, 255),5)

    return last_frame_time

def setupSocket():
    # Socket Create
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = socket.gethostbyname('localhost')
    print('HOST IP:', host_ip)
    port = 65432
    socket_address = (host_ip, port)

    # Socket Bind
    server_socket.bind(socket_address)

    # Socket Listen
    server_socket.listen(5)
    print("LISTENING AT:", socket_address)

    client_socket, addr = server_socket.accept()
    print('GOT CONNECTION FROM:', addr)

    return client_socket

# Sender Code
def send_video():

    last_frame_time = 0

    client_socket = setupSocket()
   
    if client_socket:
        vid = cv2.VideoCapture(VIDEO_PATH)
        print("fps:", vid.get(cv2.CAP_PROP_FPS))

        while (vid.isOpened()):
            _, frame = vid.read()
            a = pickle.dumps(frame)
            message = struct.pack("Q", len(a)) + a
            client_socket.sendall(message)

            last_frame_time = calculateFps(last_frame_time, frame)
            
            cv2.imshow('TRANSMITTING VIDEO', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                client_socket.close()

send_video()