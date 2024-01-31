from flask import Flask, render_template, Response, request
import cv2
import os
from counting_vehicles import *

app = Flask(__name__)

FRAME_FOR_MASK_CREATION = 5
COUNT_LINE_YPOS = 0

def process_video(video_capture_index, frame_rate="0"):

    videoCapture = cv2.VideoCapture(video_capture_index)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    vehicleCounter = 0
    
    # extract frame to create the mask
    frameForMask = []    
    for _ in range(FRAME_FOR_MASK_CREATION):
        ret, frame = videoCapture.read()
        if ret != False:
            frameForMask.append(frame)

    # extract the mask and set the counting line position
    ret, frame = videoCapture.read()
    if ret:
        mask = cropStreet(frameForMask)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # set the counting line position
        count_line_y_pos = int((frame.shape[0] * 4 / 5))

    last_frame_time = 0
    framesSent = 0
    
    while True:
        ret, frame = videoCapture.read()
        if ret == False:
            break

        masked_frame = cv2.bitwise_and(frame, mask)
        gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        filteredImage = extractBgAndFilter(masked_frame, bg_subtractor)
        contours, _ = cv2.findContours(
            filteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        detectedVehicles, boundingBoxes = drawBoundingBoxes(contours, frame)

        vehicleCounter = countVehicles(frame, detectedVehicles, vehicleCounter, count_line_y_pos)

        frame = detectVehiclesClass(filteredImage, frame, boundingBoxes)

        # draw counter
        cv2.putText(
            frame,
            "Vehicle detected: " + str(vehicleCounter),
            (70, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5,
        )

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        draw_flow(gray_frame, frame, flow, boundingBoxes)
        prev_gray = gray_frame

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        framesSent += 1

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        # cv2.imshow("Vehicles flows", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break
        if framesSent % 50 == 0:
            last_frame_time, currentFps = calculateFps(last_frame_time,50)
            print(f"Vehicle detected: {vehicleCounter}, Current FPS:{currentFps}")
        
        if frame_rate != 0:
            time.sleep(1/int(frame_rate*2))

@app.route('/')
def index():

    video_captures= [{"name": 0, "frame_rates": ["max"]}]
    for video_path in os.listdir('assets'):
        cap = cv2.VideoCapture("assets/" + video_path)
        expected_fps = int(cap.get(cv2.CAP_PROP_FPS))
        video_captures.append({"name": video_path, "frame_rates": ["max", expected_fps, 20, 10]})

    print(video_captures)

    return render_template('index.html', video_captures=video_captures)

@app.route('/watch_stream', methods=['POST'])
def watch_stream():
    video_name = request.form['video_name']
    frame_rate = request.form['frame_rate']

    return render_template('video_viewer.html', video_name=video_name, frame_rate=frame_rate)

@app.route('/video/<string:video_name>/<string:frame_rate>')
def video(video_name, frame_rate):
    video_name = "assets/" + video_name if video_name != "0" else 0
    frame_rate = int(frame_rate) if frame_rate != "max" else 0

    return Response(process_video(video_name, frame_rate), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream_webcam')
def vlc_stream():
    return Response(process_video("assets/video.mp4", 30), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
