import socket
import cv2
import pickle
import struct
import time

def calculateFps(last_frame_time, framesNumber) -> tuple[float, str]:
    """
    Calculate the fps and put it on the frame

    Args:
        prev_frame_time (float): the previous frame time
        framesNumber (int): the number of frames to calculate the fps

    Returns:
        last_frame_time (float): the last frame time
        fps (str): the fps
    """

    new_frame_time = time.time()
    fps = framesNumber/(new_frame_time-last_frame_time) 
    last_frame_time = new_frame_time 

    fps = int(fps) 
    fps = str(fps) 

    return last_frame_time, fps

# Receiver Code
def receive_video():

    # create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '127.0.0.1'  # paste your server ip address here
    port = 65432
    client_socket.connect((host_ip, port))  # a tuple
    data = b""
    payload_size = struct.calcsize("Q")

    last_frame_time = 0
    received_frames = 0
    frames_to_display = []

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4 * 1024)  # 4K
            if not packet: break
            data += packet
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4 * 1024)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        frames = pickle.loads(frame_data)

        frames_to_display.extend(frames)
        
        received_frames += len(frames_to_display)
        if received_frames % 50 == 0:
            last_frame_time, fps = calculateFps(last_frame_time, 50)
            print(f"Current FPS: {fps}")

        for frame in frames_to_display:
            cv2.imshow("RECEIVING VIDEO", frame)

        frames_to_display = []

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            client_socket.close()
            break
    client_socket.close()

receive_video()