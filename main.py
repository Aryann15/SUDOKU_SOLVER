import cv2 as cv

frame = cv.VideoCapture(0)

frame.set(3,640)
frame.set(4,480)

while True:
    success , img = frame.read()
    cv.imshow('webcam',img)
    if cv.waitKey(1) & 0xff == ord('q'):
        break