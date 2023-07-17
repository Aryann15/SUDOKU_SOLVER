import cv2
import cv2 as cv
import numpy as np
import  matplotlib.pyplot as plt
frame = cv.VideoCapture(0)

frame.set(3,640)
frame.set(4,480)



def get_contours(img,original_img):
    contours,hierarchy = cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 30000:
            cv.drawContours(original_img ,cnt,-1,(0,255,0),2)
            peri = cv.arcLength(cnt,True)
            approx = cv.approxPolyDP(cnt,0.02*peri,True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width,height = 900,900

            pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
            pts2 = np.float32([[0,0], [width,0], [0,height], [width,height]])

            matrix = cv.getPerspectiveTransform(pts1,pts2)
            img_perspective= cv.warpPerspective(original_img,matrix,(width,height))
            contours= cv.cvtColor(img_perspective,cv.COLOR_BGR2GRAY)
            # cv.imshow('contour', contours)
            for x in range(0,900):
                for y in range (0,900):
                    if contours[x][y]<100:
                        contours[x][y] = 0
                    else:
                         contours[x][y]= 255
            cv.imshow('contour',contours)


def pick_images(grid, img):
    data_drop = 'data'
    crop = 20
    listnum= []
    for i in range(0,9):
        for j in range (0,9):
            if grid[i][j] not in listnum:
                print(grid[i][j])
                listnum.append(grid[i][j])
                J = j+1
                I = i+1
                cell = img[I*100-100 +crop : I*100 -crop, J*100-100 +crop: J*100 -crop]

                cv.imwrite(data_drop+ str(grid[i][j]) + '//IMG_{}.png'.format((i+1)*(j+1),cell)