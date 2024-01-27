import numpy as np
import cv2
import os
import time
import socket
import pickle
import struct

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

COUNT_LINE_YPOS = 0
CONTOUR_WIDTH = (70, 300)
CONTOUR_HEIGHT = (70, 300)
OFFSET_FOR_DETECTION = 8

FRAME_FOR_MASK_CREATION = 5
BATCH_SIZE = 3

def setupSocket() -> socket.socket:
    """
    Setup the socket to wait for a client to stream the video

    Returns:
        client_socket (socket.socket): the client socket
    """
    # Socket Create
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    host_ip = '127.0.0.1'
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


def extend_line(line, imageWidth)-> tuple[int, int, int, int]:
    """
    Extend a line across the entire image while maintaining its direction.

    Args:
        line (tuple): the line
        imageWidth (int): the image width

    Returns:
        extended_x1 (int): the x coordinate of the first point of the extended line
        extended_y1 (int): the y coordinate of the first point of the extended line
        extended_x2 (int): the x coordinate of the second point of the extended line
        extended_y2 (int): the y coordinate of the second point of the extended line
    """
    x1, y1, x2, y2 = line
    slope = (y2 - y1) / (
        x2 - x1 + 1e-5
    )  # Adding a small value to avoid division by zero

    # Extend the line to the image borders
    extended_x1 = 0
    extended_y1 = int(y1 - slope * (x1 - extended_x1))

    extended_x2 = imageWidth - 1
    extended_y2 = int(y2 - slope * (x2 - extended_x2))

    return extended_x1, extended_y1, extended_x2, extended_y2


def find_intersection_point(line1, line2)-> tuple[float, float]:
    """
    Find the intersection point between two lines defined by their coordinates.
    Each line is represented by two points (x1, y1), (x2, y2).

    Args:
        line1 (tuple): the first line
        line2 (tuple): the second line

    Returns:
        x_intersect (float): the x coordinate of the intersection point
        y_intersect (float): the y coordinate of the intersection point
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate slopes and intercepts for each line
    m1 = (y2 - y1) / (x2 - x1 + 1e-5)  # Adding a small value to avoid division by zero
    b1 = y1 - m1 * x1

    m2 = (y4 - y3) / (x4 - x3 + 1e-5)  # Adding a small value to avoid division by zero
    b2 = y3 - m2 * x3

    # Calculate intersection point
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1

    return x_intersect, y_intersect


def cropStreet(frames) -> np.ndarray:
    """
    Create a mask for cropping the street

    Args:
        frames (list): a list of frames

    Returns:
        mask (np.ndarray): the mask
    """
    for frame in frames:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
        pixel_values = image.reshape(-1, 3)

        # Convert to float type to apply kmeans algorithm
        pixel_values = np.float32(pixel_values)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
        k = 3
        retval, labels, centers = cv2.kmeans(
            pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )

        # Segmenting the original image with kmeans results
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))

        edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 30, minLineLength=200, maxLineGap=5
        )
        if lines is None:
            print("No lines detected")
            return segmented_image

        lines = lines.reshape(-1, 4)  # remove redundant dimensions
        validLines = ([], [])

        for line in lines:
            extended_line = extend_line(line, image.shape[1])
            x1, y1, x2, y2 = extended_line

            # discard lines mostly horizontal or vertical
            slope = (y2 - y1) / (x2 - x1 + 1e-5)  # avoid division by zero
            if abs(slope) < 0.2 or abs(slope) > 2.0:
                continue

            if y1 > y2:  # left line
                validLines[0].append(extended_line)
            else:  # right line
                validLines[1].append(extended_line)

    leftmostLine = (0, 0, 0, 0)
    rightmostLine = (0, 0, 0, 0)

    if len(validLines[0]) > 0:
        leftmostLine = min(validLines[0], key=lambda x: x[0])
    if len(validLines[1]) > 0:
        rightmostLine = max(validLines[1], key=lambda x: x[0])

    # draw the lines
    cv2.line(
        segmented_image,
        (leftmostLine[0], leftmostLine[1]),
        (leftmostLine[2], leftmostLine[3]),
        (255, 0, 0),
        3,
    )
    cv2.line(
        segmented_image,
        (rightmostLine[0], rightmostLine[1]),
        (rightmostLine[2], rightmostLine[3]),
        (255, 255, 0),
        3,
    )
    intersectingPoint = find_intersection_point(leftmostLine, rightmostLine)

    # create a mask to crop the street
    bottomLeft = (0, image.shape[0])
    bottomRight = (image.shape[1], image.shape[0])
    mask = np.zeros(frame.shape, dtype=np.uint8)
    roi_corners = np.array(
        [
            [
                bottomLeft,
                (leftmostLine[0], leftmostLine[1]),
                intersectingPoint,
                (rightmostLine[2], rightmostLine[3]),
                bottomRight,
            ]
        ],
        dtype=np.int32,
    )
    channel_count = frame.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)

    return mask


def extractBgAndFilter(frame, bg_subtractor) -> np.ndarray:
    """
    Extract background and filter a frame

    Args:
        frame (np.ndarray): the frame
        bg_subtractor (cv2.BackgroundSubtractor): the background subtractor

    Returns:
        dilated (np.ndarray): the filtered frame
    """

    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(greyFrame, (15, 15), 0)

    img_sub = bg_subtractor.apply(blurredFrame)
    dilatatedFrame = cv2.dilate(img_sub, np.ones((5, 5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilatatedFrame, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return dilated


def filterAndUnifyLines(lines, boundingBoxes) -> np.ndarray:
    """
    Filter lines that are inside bounding boxes and return a single average of all of them for each box

    Args:
        lines (np.ndarray): the lines
        boundingBoxes (list): a list of bounding boxes

    Returns:
        filteredLines (np.ndarray): the filtered lines
    """

    linesInBoundingBoxes = {}
    filteredLines = np.empty((0, 2, 2), dtype=np.int32)

    # for each bounding box find the lines that are inside it
    for i, (x, y, w, h) in enumerate(boundingBoxes):
        linesInBoundingBoxes[i] = []
        for line in lines:
            x1, y1 = line[0]
            x2, y2 = line[1]
            if x1 > x and x2 < (x + w) and y1 > y and y2 < (y + h):
                linesInBoundingBoxes[i].append(line)

    # for each bounding identified by a dict key find the average of all the points
    for boxId in linesInBoundingBoxes.keys():
        avgStartPoint = np.zeros(2)
        avgStopPoint = np.zeros(2)

        for line in linesInBoundingBoxes[boxId]:
            avgStartPoint += line[0]
            avgStopPoint += line[1]

        avgStartPoint = avgStartPoint / len(linesInBoundingBoxes[boxId])
        avgStopPoint = avgStopPoint / len(linesInBoundingBoxes[boxId])
        filteredLines = np.vstack(
            [filteredLines, np.array([[avgStartPoint, avgStopPoint]]).astype(int)]
        )

    return filteredLines


def scale_vector(p1, p2, scaling_factor):
    """
    Scale a vector defined by two points (p1 and p2) by a scaling factor.

    Args:
        p1 (tuple): the first point
        p2 (tuple): the second point
        scaling_factor (int): the scaling factor

    Returns:
        new_x (int): the new x coordinate
        new_y (int): the new y coordinate
    """
    x1, y1 = p1
    x2, y2 = p2
    new_x = x1 + scaling_factor * (x2 - x1)
    new_y = y1 + scaling_factor * (y2 - y1)

    return int(new_x), int(new_y)


def draw_flow(img, img_bgr, flow, boundingBoxes, step=16):
    """
    Draw the optical flow vectors on the given image.

    Args:
        img (np.ndarray): the image
        img_bgr (np.ndarray): the image in BGR format
        flow (np.ndarray): the optical flow
        boundingBoxes (list): a list of bounding boxes
        step (int): the step for the lines

    """

    h, w = img.shape[:2]
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    lines = filterAndUnifyLines(lines, boundingBoxes)

    for line in lines:
        start_point = tuple(line[1])
        end_point = tuple(line[0])
        end_point_rescaled = scale_vector(start_point, end_point, 5)
        cv2.arrowedLine(
            img_bgr, start_point, end_point_rescaled, (0, 255, 0), 2, tipLength=0.3
        )



def centerCoordinates(x, y, w, h) -> tuple:
    """
    Calculate center coordinates of a bounding box

    Args:
        x (int): x coordinate of the bounding box
        y (int): y coordinate of the bounding box
        w (int): width of the bounding box
        h (int): height of the bounding box

    Returns:
        cx (int): x coordinate of the center
        cy (int): y coordinate of the center
    """

    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return cx, cy


def drawBoundingBoxes(contours, frame) -> list:
    """
    Draw bounding boxes around detected vehicles and return their center coordinates

    Args:
        contours (list): a list of contours
        frame (np.ndarray): the frame

    Returns:
        detectedVehicles (list): a list of detected vehicles
        boundingBoxes (list): a list of bounding boxes
    """

    detectedVehicles = []
    boundingBoxes = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (
            (w >= CONTOUR_WIDTH[0])
            and (h >= CONTOUR_HEIGHT[0])
            and (w <= CONTOUR_WIDTH[1])
            and (h <= CONTOUR_HEIGHT[1])
        )

        if not contour_valid:
            continue

        # draw bounding boxes
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        boundingBoxes.append((x, y, w, h))

        center = centerCoordinates(x, y, w, h)
        detectedVehicles.append(center)

        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    return detectedVehicles, boundingBoxes


def countVehicles(frame, detectedVehicles, vehicleCounter) -> int:
    """
    Count vehicles that cross the counting line with a margin of error

    Args:
        frame (np.ndarray): the frame
        detectedVehicles (list): a list of detected vehicles
        vehicleCounter (int): the vehicle counter

    Returns:
        vehicleCounter (int): the vehicle counter
    """

    for x, y in detectedVehicles:
        if (
            (COUNT_LINE_YPOS - OFFSET_FOR_DETECTION)
            <= y
            <= (COUNT_LINE_YPOS + OFFSET_FOR_DETECTION)
        ):
            vehicleCounter += 1
            cv2.line(
                frame, (25, COUNT_LINE_YPOS), (1200, COUNT_LINE_YPOS), (0, 127, 255), 3
            )
            detectedVehicles.remove((x, y))

            print(f"Vehicle detected: {vehicleCounter}")

    return vehicleCounter


def detectVehiclesClass(filteredImage, frame, boundingBoxes) -> np.ndarray:
    """Detect the vehicles class (car or truck) and print it on the frame

    Args:
        filteredImage (np.ndarray): the filtered image
        frame (np.ndarray): the frame
        boundingBoxes (list): a list of bounding boxes

    """

    # crop image on the bounding boxes
    for i, (x, y, w, h) in enumerate(boundingBoxes):
        crop = filteredImage[y : y + h, x : x + w]
        # count white pixels
        whitePixels = cv2.countNonZero(crop)

        # calculate the percentage of white pixels
        percentage = whitePixels / (w * h)
        boundingBoxSize = w * h

        score = calculateScore(percentage, boundingBoxSize)

        # put the label on the frame
        if score > 0.5:
            cv2.putText(
                frame,
                "Truck",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Car",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )


def calculateScore(percentage_white_pixels, bounding_box_size) -> float:
    """
    Calculate a score based on the percentage of white pixels and the bounding box size

    Args:
        percentage_white_pixels (float): the percentage of white pixels
        bounding_box_size (int): the bounding box size

    Returns:
        score (float): the score
    """
    # Define maximum bounding box size (not precise)
    max_bounding_box_size = 20000
    # Define weights for combining normalized percentage and size
    weight_percentage = 0.7
    weight_size = 0.3

    # Normalize the values to ensure they are on a similar scale
    normalized_percentage = percentage_white_pixels / 100.0  # Normalize to [0, 1]
    normalized_size = bounding_box_size / max_bounding_box_size  # Normalize to [0, 1]

    # Combine normalized percentage and size using weighted sum
    score = (weight_percentage * normalized_percentage) + (
        weight_size * normalized_size
    )

    return score

def send_batch(frames, client_socket):
    # Serialize and send multiple frames in a batch    
    data = pickle.dumps(frames)
    message = struct.pack("Q", len(data)) + data
    client_socket.sendall(message)

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

def process_video(videoCapture):
    """
    Process video frame by frame displaying the results
    """

    fps = videoCapture.get(cv2.CAP_PROP_FPS)

    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    vehicleCounter = 0

    frameForMask = []
    framesSent = 0

    # extract frame to create the mask
    for _ in range(FRAME_FOR_MASK_CREATION):
        ret, frame = videoCapture.read()
        if ret != False:
            frameForMask.append(frame)

    ret, frame = videoCapture.read()

    if ret:
        mask = cropStreet(frameForMask)
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # set the counting line position
        global COUNT_LINE_YPOS
        COUNT_LINE_YPOS = int((frame.shape[0] * 4 / 5))

    last_frame_time = 0
    client_socket = setupSocket()

    if not client_socket:
        return
    
    framesToSend = []
    
    while True:
        ret, frame = videoCapture.read()
        if ret == False:
            break

        # masked_frame = cv2.bitwise_and(frame, mask)
        # gray_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # filteredImage = extractBgAndFilter(masked_frame, bg_subtractor)
        # contours, _ = cv2.findContours(
        #     filteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        # )
        # detectedVehicles, boundingBoxes = drawBoundingBoxes(contours, frame)

        # vehicleCounter = countVehicles(frame, detectedVehicles, vehicleCounter)

        # detectVehiclesClass(filteredImage, frame, boundingBoxes)

        # # draw counter
        # cv2.putText(
        #     frame,
        #     "Vehicle detected: " + str(vehicleCounter),
        #     (70, 70),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     2,
        #     (0, 0, 255),
        #     5,
        # )

        # flow = cv2.calcOpticalFlowFarneback(
        #     prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        # )
        # draw_flow(gray_frame, frame, flow, boundingBoxes)
        # prev_gray = gray_frame
        
        # send the frame to the client
        framesToSend.append(frame)
        if len(framesToSend) == BATCH_SIZE:
            send_batch(framesToSend, client_socket)
            framesSent += BATCH_SIZE
            framesToSend = []

        if framesSent % 50 == 0:
            last_frame_time, currentFps = calculateFps(last_frame_time,50)
            print(f"Current FPS:{currentFps}")

        cv2.imshow("Vehicles flows", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            client_socket.close()
            break
    client_socket.close()


def main():
    videoCapture = cv2.VideoCapture(VIDEO_PATH)
    print("Expected frame rate:", videoCapture.get(cv2.CAP_PROP_FPS))
    print("Frame number:", videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    process_video(videoCapture)


if __name__ == "__main__":
    main()
