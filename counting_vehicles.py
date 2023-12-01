import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video2.mp4")

COUNT_LINE_POS = 0
CONTOUR_WIDTH = (70, 300)
CONTOUR_HEIGHT = (70, 300)
OFFSET_FOR_DETECTION = 8

def extend_line(line, image_shape):
    """
    Extend a line across the entire image while maintaining its direction.
    """
    x1, y1, x2, y2 = line

    # Calculate the slope of the line
    slope = (y2 - y1) / (x2 - x1 + 1e-5)  # Adding a small value to avoid division by zero

    # Extend the line to the image borders
    extended_x1 = 0
    extended_y1 = int(y1 - slope * (x1 - extended_x1))

    extended_x2 = image_shape[1] - 1
    extended_y2 = int(y2 - slope * (x2 - extended_x2))

    return [extended_x1, extended_y1, extended_x2, extended_y2]

def cropStreet(frame):
    """
    Create a mask for cropping the street
    """
    # Change color to RGB (from BGR)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    pixel_vals = image.reshape((-1,3)) # numpy reshape operation -1 unspecified 

    # Convert to float type only for supporting cv2.kmean
    pixel_vals = np.float32(pixel_vals)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    k = 3
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((image.shape))
    edges = cv2.Canny(segmented_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=200, maxLineGap=5)
    if lines is not None:
        for line in lines:
            extended_line = extend_line(line[0], image.shape)
            x1, y1, x2, y2 = extended_line
            slope = (y2 - y1) / (x2 - x1 + 1e-5) # avoid division by zero
            # coeff tra 1 e 2.5/3
            if y1 > y2:         # left line
                if -2 < slope < -0.2:
                    cv2.line(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:               # right line
                if 0.2 < slope < 2:    
                    cv2.line(segmented_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
    # Mask creation below        
    
    # # create a mask to crop the street
    # mask = np.zeros(frame.shape, dtype=np.uint8)
    # roi_corners = np.array([[(meanRightLine[0], meanRightLine[1]), (meanRightLine[2], meanRightLine[3]), (0,frame.shape[0])]], dtype=np.int32)
    # channel_count = frame.shape[2]
    # ignore_mask_color = (255,) * channel_count
    # cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    return segmented_image


def extractBgAndFilter(frame, bg_subtractor) -> np.ndarray:
    """
    Extract background and filter a frame
    """

    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(greyFrame, (15, 15), 0)

    img_sub = bg_subtractor.apply(blurredFrame)
    dilatatedFrame = cv2.dilate(img_sub, np.ones((5, 5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(dilatatedFrame, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    return dilated


# TODO - improve this function

def filterAndUnifyLines(lines, boundingBoxes):
    """
    Filter lines that are inside bounding boxes and return a single average of all of them for each box
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

    Parameters:
    - p1: Tuple representing the coordinates (x, y) of the first point.
    - p2: Tuple representing the coordinates (x, y) of the second point.
    - scaling_factor: The factor by which to scale the vector.

    Returns:
    - Tuple representing the coordinates of the new endpoint after scaling.
    """
    x1, y1 = p1
    x2, y2 = p2
    new_x = x1 + scaling_factor * (x2 - x1)
    new_y = y1 + scaling_factor * (y2 - y1)

    return int(new_x), int(new_y)


def draw_flow(img, img_bgr, flow, boundingBoxes, step=16):
    """
    Draw the optical flow vectors on the given image.
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

    return img_bgr


def centerCoordinates(x, y, w, h) -> tuple:
    """
    Calculate center coordinates of a bounding box
    """

    x1 = int(w / 2)
    y1 = int(h / 2)

    cx = x + x1
    cy = y + y1

    return cx, cy


def drawBoundingBoxes(contours, frame) -> list:
    """
    Draw bounding boxes around detected vehicles and return their center coordinates
    """

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
    """
    Count vehicles that cross the counting line with a margin of error
    """

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
    """
    Process video frame by frame and detect vehicles counting them
    """

    bg_subtractor = cv2.createBackgroundSubtractorKNN(history=100, detectShadows=False)
    vehicleCounter = 0
    ret, prev_frame = videoCapture.read()
    if ret:
        # mask = cropStreet(prev_frame)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = videoCapture.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cropStreet(frame)
        # masked_frame = cv2.bitwise_and(frame, mask)
        # global COUNT_LINE_POS
        # COUNT_LINE_POS = int((frame.shape[0] * 4 / 5))

        # if ret == False:
        #     break

        # filteredImage = extractBgAndFilter(masked_frame, bg_subtractor)

        # contours, _ = cv2.findContours(
        #     filteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        # )

        # # draw counting line
        # cv2.line(frame, (25, COUNT_LINE_POS), (1200, COUNT_LINE_POS), (255, 127, 0), 3)

        # detectedVehicles, boundingBoxes = drawBoundingBoxes(contours, frame)

        # vehicleCounter = countVehicles(frame, detectedVehicles, vehicleCounter)
        # cv2.putText(
        #     frame,
        #     "Vehicle detected: " + str(vehicleCounter),
        #     (450, 70),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     2,
        #     (0, 0, 255),
        #     5,
        # )

        # flow = cv2.calcOpticalFlowFarneback(
        #     prev_gray, gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
        # )
        # frame_with_flow = draw_flow(gray_frame, frame, flow, boundingBoxes)
        cv2.imshow("Vehicles flows", frame)
        prev_gray = gray_frame
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


def main():
    videoCapture = cv2.VideoCapture(VIDEO_PATH)
    process_video(videoCapture)


if __name__ == "__main__":
    main()
