import socket
import cv2
import pickle
import struct
import time

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

# Receiver Code
def receive_video():

    # create socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = 'localhost'  # paste your server ip address here
    port = 65432
    client_socket.connect((host_ip, port))  # a tuple
    data = b""
    payload_size = struct.calcsize("Q")

    last_frame_time = 0

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
        frame = pickle.loads(frame_data)

        last_frame_time = calculateFps(last_frame_time, frame)

        cv2.imshow("RECEIVING VIDEO", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    client_socket.close()

receive_video()