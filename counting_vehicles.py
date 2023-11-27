import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

COUNT_LINE_POS = 0
CONTOUR_WIDTH = (70, 300)
CONTOUR_HEIGHT = (70, 300)
OFFSET_FOR_DETECTION = 8


def extractBgAndFilter(frame, bg_subtractor) -> np.ndarray:
    """Extract background and filter a frame"""

    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(greyFrame, (15, 15), 0)

    img_sub = bg_subtractor.apply(blurredFrame)
    dilatatedFrame = cv2.dilate(img_sub, np.ones((5, 5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilatatedFrame, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return dilated


# TODO - improve this function


def filterLines(lines, boundingBoxes):
    """Filter lines that are inside bounding boxes and return a single average of all of them for each box"""

    linesInBoundingBoxes = {}
    filteredLines = np.empty((0, 2, 2), dtype=np.int32)
    for line in lines:
        x1, y1 = line[0]
        x2, y2 = line[1]
        for i, (x, y, w, h) in enumerate(boundingBoxes):
            if x1 > x and x2 < (x + w) and y1 > y and y2 < (y + h):
                linesInBoundingBoxes[i].append(line)
                
    print(linesInBoundingBoxes)
    # for each bounding identified by a dict key find the average of all the points
    for boxId in linesInBoundingBoxes.keys():
        avgStartPoint = np.zeros(2)
        avgStopPoint = np.zeros(2)

        for line in linesInBoundingBoxes[boxId]:
            avgStartPoint += line[0]
            avgStopPoint += line[1]

        avgStartPoint = avgStartPoint / len(linesInBoundingBoxes[boxId])
        avgStopPoint = avgStopPoint / len(linesInBoundingBoxes[boxId])
        #assign to filtered lines such us the resulting shape is (?, 2,2)
        # avgLine = np.array([avgStartPoint, avgStopPoint])
        # filteredLines = np.vstack([filteredLines, avgLine])
        
        

    return filteredLines


def draw_flow(img, flow, boundingBoxes, step=16):
    h, w = img.shape[:2]
    # TODO - improve this function (compute only the one that are inside the bounding boxes)
    y, x = np.mgrid[step / 2 : h : step, step / 2 : w : step].reshape(2, -1).astype(int)
    fx, fy = flow[y, x].T

    lines = np.vstack([x, y, x - fx, y - fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)

    lines = filterLines(lines, boundingBoxes)

    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    return img_bgr


def centerCoordinates(x, y, w, h) -> tuple:
    """Calculate center coordinates of a bounding box"""

    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return cx, cy


def drawBoundingBoxes(contours, frame) -> list:
    """Draw bounding boxes around detected vehicles and return their center coordinates"""

    detectedVehicles = []
    boundingBoxes = []

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= CONTOUR_WIDTH[0]) and (
            h >= CONTOUR_HEIGHT[0]
        )  # and w <= (CONTOUR_WIDTH[1])  and (h <= CONTOUR_HEIGHT[1])

        if not contour_valid:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        boundingBoxes.append((x, y, w, h))

        center = centerCoordinates(x, y, w, h)
        detectedVehicles.append(center)

        cv2.circle(frame, center, 4, (0, 0, 255), -1)

    return detectedVehicles, boundingBoxes


def countVehicles(frame, detectedVehicles, vehicleCounter) -> int:
    """Count vehicles that cross the counting line with a margin of error"""

    for x, y in detectedVehicles:
        if y < (COUNT_LINE_POS + OFFSET_FOR_DETECTION) and y > (
            COUNT_LINE_POS - OFFSET_FOR_DETECTION
        ):
            vehicleCounter += 1
            cv2.line(
                frame, (25, COUNT_LINE_POS), (1200, COUNT_LINE_POS), (0, 127, 255), 3
            )
            detectedVehicles.remove((x, y))

            print(f"Vehicle detected: {vehicleCounter}")

    return vehicleCounter


def process_video(videoCapture):
    """Process video frame by frame and detect vehicles counting them"""

    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    vehicleCounter = 0
    ret, prev_frame = videoCapture.read()
    if ret:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = videoCapture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        global COUNT_LINE_POS
        COUNT_LINE_POS = int((frame.shape[0] * 4 / 5))

        if ret == False:
            break

        # TODO - improve background subtraction
        filteredImage = extractBgAndFilter(frame, bg_subtractor)

        contours, _ = cv2.findContours(
            filteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # draw counting line
        cv2.line(frame, (25, COUNT_LINE_POS), (1200, COUNT_LINE_POS), (255, 127, 0), 3)

        detectedVehicles, boundingBoxes = drawBoundingBoxes(contours, frame)

        vehicleCounter = countVehicles(frame, detectedVehicles, vehicleCounter)
        cv2.putText(
            frame,
            "Vehicle detected: " + str(vehicleCounter),
            (450, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            5,
        )

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        prev_gray = gray_frame

        frame_with_flow = draw_flow(gray_frame, flow, boundingBoxes)
        # cv2.imshow("Vehicles counter", frame_with_flow)
        cv2.imshow("Vehicles counter", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break


def main():
    videoCapture = cv2.VideoCapture(VIDEO_PATH)
    process_video(videoCapture)


if __name__ == "__main__":
    main()
