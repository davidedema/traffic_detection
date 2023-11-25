import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

def draw_bounding_boxes(contours):
    bounding_boxes = []

    for contour in contours:
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Filter out small contours (noise)
        if area > 800:  # Adjust the threshold as needed
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, x + w, y + h))
    
    return bounding_boxes

def remove_overlapping_boxes(bounding_boxes):
    non_overlapping_boxes = []
    
    for i in range(len(bounding_boxes)):
        x1_i, y1_i, x2_i, y2_i = bounding_boxes[i]
        overlapping = False

        for j in range(len(bounding_boxes)):
            if i != j:
                x1_j, y1_j, x2_j, y2_j = bounding_boxes[j]

                # Check if the bounding boxes overlap
                if( x1_i < x2_j and x2_i > x1_j and y1_i < y2_j and y2_i > y1_j):
                    overlapping = True
                    break
        
        # If the bounding box does not overlap, add it to the list
        if not overlapping:
            non_overlapping_boxes.append(bounding_boxes[i])
    
    return non_overlapping_boxes

def getCentroid(x1, y1, x2, y2):
    centroid_x = int((x1 + x2) / 2)
    centroid_y = int((y1 + y2) / 2)

    return centroid_x, centroid_y

def detect_direction(current_centroids, prev_centroids, frame):
    try:
        for i in range(len(current_centroids)):
            if len(prev_centroids) > 0:
                if current_centroids[i][1] < prev_centroids[i][1]:
                    cv2.putText(
                        frame,
                        "UP",
                        (current_centroids[i][0], current_centroids[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                elif current_centroids[i][1] > prev_centroids[i][1]:
                    cv2.putText(
                        frame,
                        "DOWN",
                        (current_centroids[i][0], current_centroids[i][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
    except:
        print("Index error")

def process_video(videoCapture):
    
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=30)
    prev_centroids = []

    while videoCapture.isOpened():
        ret, frame = videoCapture.read()
        blur = cv2.GaussianBlur(frame, (15, 15), 0)

        fgmask = fgbg.apply(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))

        if ret == False:
            break
        
        # Find contours of the objects
        contours, _ = cv2.findContours(
            fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        bounding_boxes = draw_bounding_boxes(contours)        

        non_overlapping_boxes = remove_overlapping_boxes(bounding_boxes)

        # Draw non-overlapping bounding rectangles and detect direction
        current_centroids = []
        for box in non_overlapping_boxes:
            x1, y1, x2, y2 = box
            centroid = getCentroid(x1, y1, x2, y2)
            current_centroids.append(centroid)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.circle(frame, centroid, 3, (0, 255, 0), -1)
        
        # Detect the direction (UP or DOWN) of the cars
        detect_direction(current_centroids, prev_centroids, frame)
        prev_centroids = current_centroids

        cv2.imshow("frame", frame)

        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    videoCapture.release()
    cv2.destroyAllWindows()

def main():
    videoCapture = cv2.VideoCapture(VIDEO_PATH)
    process_video(videoCapture)

if __name__ == "__main__":
    main()
