import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

COUNT_LINE_POS = 550
MIN_CONTOUR_WIDTH = 70
MIN_CONTOUR_HEIGHT = 70
OFFSET_FOR_DETECTION = 6

def extractBgAndFilter(frame, bg_subtractor):

    greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurredFrame = cv2.GaussianBlur(greyFrame, (15,15), 0)

    img_sub = bg_subtractor.apply(blurredFrame)
    dilatatedFrame = cv2.dilate(img_sub, np.ones((5,5)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    dilated = cv2.morphologyEx(dilatatedFrame, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Dilated", img_sub)

    return dilated

def centerCoordinates(x, y, w, h):
    x1 = int(w/2)
    y1 = int(h/2)

    cx = x + x1
    cy = y + y1

    return cx, cy

def drawBoundingBoxes(contours, frame):
    detectedVehicles = []

    for (i, c) in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)
        contour_valid = (w >= MIN_CONTOUR_WIDTH) and (h >= MIN_CONTOUR_HEIGHT)

        if not contour_valid:
            continue

        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        center = centerCoordinates(x, y, w, h)
        detectedVehicles.append(center)

        cv2.circle(frame, center, 4, (0,0,255), -1)

    return detectedVehicles

def countVehicles(frame, detectedVehicles, vehicleCounter):

    for (i, (x,y)) in enumerate(detectedVehicles):
        if y < (COUNT_LINE_POS + OFFSET_FOR_DETECTION) and y > (COUNT_LINE_POS - OFFSET_FOR_DETECTION):
            vehicleCounter += 1
            cv2.line(frame, (25, COUNT_LINE_POS), (1200, COUNT_LINE_POS), (0,127,255), 3)
            detectedVehicles.remove((x,y))

            print(f"Vehicle detected: {vehicleCounter}")

    return vehicleCounter

def process_video(videoCapture):

    bg_subtractor = cv2.createBackgroundSubtractorKNN()
    vehicleCounter = 0

    while True:
        ret, frame = videoCapture.read()

        if ret == False:
            break

        filteredImage = extractBgAndFilter(frame, bg_subtractor)

        contours, _ = cv2.findContours(filteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #draw counting line
        cv2.line(frame, (25, COUNT_LINE_POS), (1200, COUNT_LINE_POS), (255,127,0), 3)

        detectedVehicles = drawBoundingBoxes(contours, frame)

        vehicleCounter = countVehicles(frame, detectedVehicles, vehicleCounter)
        cv2.putText(frame, "Vehicle detected: " + str(vehicleCounter), (450,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 5)

        cv2.imshow("Detection2", filteredImage)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

def main():
    videoCapture = cv2.VideoCapture(VIDEO_PATH)
    process_video(videoCapture)

if __name__ == "__main__":
    main()