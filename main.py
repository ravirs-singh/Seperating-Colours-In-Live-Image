import cv2
import numpy as np


def nothing(x):
    return ()


cap = cv2.VideoCapture(0)

cv2.namedWindow("Tracker")

cv2.createTrackbar("L_Hue", "Tracker", 0, 255, nothing)
cv2.createTrackbar("L_Saturation", "Tracker", 0, 255, nothing)
cv2.createTrackbar("L_value", "Tracker", 0, 255, nothing)
cv2.createTrackbar("H_Hue", "Tracker", 255, 255, nothing)
cv2.createTrackbar("H_Saturation", "Tracker", 255, 255, nothing)
cv2.createTrackbar("H_Value", "Tracker", 255, 255, nothing)

while 1:
    # img = cv2.imread('34.jpg')
    # img = cv2.resize(img, (512, 512))

    _, img = cap.read()
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L_Hue", "Tracker")
    l_s = cv2.getTrackbarPos("L_Saturation", "Tracker")
    l_v = cv2.getTrackbarPos("L_value", "Tracker")
    u_h = cv2.getTrackbarPos("H_Hue", "Tracker")
    u_s = cv2.getTrackbarPos("H_Saturation", "Tracker")
    u_v = cv2.getTrackbarPos("H_Value", "Tracker")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    masking = cv2.inRange(img1, l_b, u_b)

    fin = cv2.bitwise_and(img, img, mask=masking)

    cv2.imshow("hsv", img1)
    cv2.imshow('Image', img)
    cv2.imshow("mask", masking)
    cv2.imshow("final", fin)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()
