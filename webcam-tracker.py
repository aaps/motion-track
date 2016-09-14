#!/usr/bin/env python

import cv2
import numpy as np

blueUpper = (130,255,255)
blueLowwer = (110,50,90)
THRESHOLD_SENSITIVITY = 25
MIN_AREA = 25
CIRCLE_SIZE = 10

cap = cv2.VideoCapture(0)

while(True):
    biggest_area = MIN_AREA
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)



    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv,blueLowwer, blueUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    # res = cv2.bitwise_and(frame,frame, mask= mask)
    retval, thresholdimage = cv2.threshold(mask,THRESHOLD_SENSITIVITY,255,cv2.THRESH_BINARY)
    # thresholdimage,contours,hierarchy = cv2.findContours(thresholdimage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours,hierarchy = cv2.findContours(thresholdimage,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        found_area = cv2.contourArea(c)
        # find the middle of largest bounding rectangle
        if found_area > biggest_area:
            motion_found = True
            biggest_area = found_area
            (x, y, w, h) = cv2.boundingRect(c)
            cx = int(x + w/2)   # put circle in middle of width
            cy = int(y + h/6)   # put circle closer to top
            cw = w
            ch = h
            # print biggest_area
            img = np.zeros((512,512,3), np.uint8)
            cv2.circle(img,(cx,cy),CIRCLE_SIZE,(0,255,0),2)
            cv2.imshow('Movement Status', img)

    # Display the resulting frame
    cv2.imshow('frame',thresholdimage)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()