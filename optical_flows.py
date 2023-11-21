import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def draw_flow(img, flow, step=16):
    
    # Get the shape of the image
    h, w = img.shape[:2]
    # Create a grid of points
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2, -1).astype(int)
    # Get the flow of the points
    fx, fy = flow[y, x].T
    # Create a grid of lines if there is a flow
    lines = np.vstack([x, y, x-fx, y-fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines)
    
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Draw the lines in the frame
    cv2.polylines(img_bgr, lines, 0, (0, 255, 0))

    # for (x1, y1), (_x2, _y2) in lines:
    #     cv2.circle(img_bgr, (x1, y1), 1, (0, 255, 0), -1)

    return img_bgr

def draw_hsv(flow):

    h, w = flow.shape[:2]
    # Get the flow of the points
    fx, fy = flow[:,:,0], flow[:,:,1]

    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx**2+fy**2)

    hsv = np.zeros((h,w,3), np.uint8)

    hsv[...,0] = ang*(180/np.pi/2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return bgr

def shi_tomasi(img): 
    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)
    # draw red color circles on all corners 
    for i in corners: 
        x, y = i.ravel() 
        cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
    return img

video = cv2.VideoCapture('/home/davide/Desktop/drones/assets/video2.mp4')

if (video.isOpened()== False): 
    print("Error opening video stream or file")

suc, prev = video.read()
prevgray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

while (video.isOpened()):
    suc, img = video.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    start = time.time()

    # Calculate the optical flow
    flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    prevgray = gray

    end = time.time()

    fps = 1/(end-start)
    if suc == True:
        # cv2.imshow('frame', shi_tomasi(img))
        cv2.imshow('flow', draw_flow(gray, flow))
        # cv2.imshow('flow HSV', draw_hsv(flow))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

  # Break the loop
    else: 
        break   

video.release()

# Closes all the frames
cv2.destroyAllWindows()

