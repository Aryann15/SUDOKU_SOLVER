import cv2
import cv2 as cv

frame = cv.VideoCapture(0)

frame.set(3,640)
frame.set(4,480)

def get_contours(img,original_img):
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 30000:
            cv.drawContours(original_img ,cnt,-1,(0,255,0),2)


while True:
    success , img = frame.read()
    gray = cv.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray,(5,5),3)
    canny = cv.Canny(blur,50,50)
    copy = img.copy()

    get_contours(canny,copy)

    cv.imshow('webcam',copy)
    if cv.waitKey(1) & 0xff == ord('q'):
        break


