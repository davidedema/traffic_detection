import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, "assets", "video.mp4")

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False, varThreshold=30)
    prev_centroids = []

    while cap.isOpened():
        ret, frame = cap.read()
        blur = cv2.GaussianBlur(frame, (15, 15), 0)

        fgmask = fgbg.apply(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))

        if ret == True:
            # Find contours of the objects
            contours, _ = cv2.findContours(
                fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Draw bounding rectangles around the objects
            bounding_boxes = []
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)

                # Filter out small contours (noise)
                if area > 800:  # Adjust the threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    bounding_boxes.append((x, y, x + w, y + h))

            # Remove overlapping boxes
            non_overlapping_boxes = []
            for i in range(len(bounding_boxes)):
                is_overlapping = False
                for j in range(len(bounding_boxes)):
                    if i != j:
                        if (
                            bounding_boxes[i][0] < bounding_boxes[j][2]
                            and bounding_boxes[i][2] > bounding_boxes[j][0]
                            and bounding_boxes[i][1] < bounding_boxes[j][3]
                            and bounding_boxes[i][3] > bounding_boxes[j][1]
                        ):
                            is_overlapping = True
                            break
                if not is_overlapping:
                    non_overlapping_boxes.append(bounding_boxes[i])

            # Draw non-overlapping bounding rectangles and detect direction
            current_centroids = []
            for box in non_overlapping_boxes:
                x1, y1, x2, y2 = box
                centroid_x = (x1 + x2) // 2
                centroid_y = (y1 + y2) // 2
                current_centroids.append((centroid_x, centroid_y))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 3, (0, 255, 0), -1)
            
            # Detect the direction (UP or DOWN) of the cars
            # FIXIT: This is not working properly
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
            prev_centroids = current_centroids

            cv2.imshow("frame", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
