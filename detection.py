import numpy as np
import cv2
import os

PATH = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(PATH, 'assets', 'video.mp4')

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while(cap.isOpened()):
        ret, frame = cap.read()
        blur = cv2.GaussianBlur(frame, (15, 15), 0)
        
        fgmask = fgbg.apply(cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY))

        if ret == True:
            # Find contours of the objects
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding rectangles around the objects
            for contour in contours:
                # Calculate the area of the contour
                area = cv2.contourArea(contour)
                
                # Filter out small contours (noise)
                if area > 1000:  # Adjust the threshold as needed
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            cv2.imshow('frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()