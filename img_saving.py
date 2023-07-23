import cv2 as cv
import numpy as np
import tensorflow as tf
import requests
import imutils

url = "http://192.168.1.2:8080/shot.jpg"

from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tensorflow.keras.utils import img_to_array
import  matplotlib.pyplot as plt

#
# frame = cv.VideoCapture(0)
#
# frame.set(3,640)
# frame.set(4,480)



def get_contours(img,original_img):
    contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 80000:
            print(area)
            cv.drawContours(original_img, cnt, -1, (0, 255, 0), 2)
            peri = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.02 * peri, True)
            ax = approx.item(0)
            ay = approx.item(1)
            bx = approx.item(2)
            by = approx.item(3)
            cx = approx.item(4)
            cy = approx.item(5)
            dx = approx.item(6)
            dy = approx.item(7)

            width, height = 630, 630

            pts1 = np.float32([[bx, by], [ax, ay], [cx, cy], [dx, dy]])
            pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

            matrix = cv.getPerspectiveTransform(pts1, pts2)
            img_perspective = cv.warpPerspective(original_img, matrix, (width, height))
            contours = cv.cvtColor(img_perspective, cv.COLOR_BGR2GRAY)
            cv.imshow('contour', contours)
            for x in range(0, 630):
                for y in range(0, 630):
                    if contours[x][y] < 100:
                        contours[x][y] = 0
                    else:
                        contours[x][y] = 255
            cv.imshow('contour', contours)
            return contours
            # classify(contours)
#
def classify(img):
    crop = 10
    digits_list= []
    for i in range(0,9):
        for j in range (0,9):
            J = j+1
            I = i+1
            cell = img[I*70-70 +crop : I*70 -crop, J*70-70 +crop: J*70 -crop]
            canny = cv.Canny(cell,50,50)
            contours,hierarchy = cv. findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

            digit = 0
            prob= 1

            for cnt in contours:
                 area = cv.contourArea(cnt)
                 if area > 5:
                    peri = cv.arcLength(cnt,True)
                    approx = cv.approxPolyDP(cnt,0.02*peri,True)
                    x,y,w,h = cv.boundingRect(approx)
                    image_rect = cell[y:y+h,x:x+w]
                    image_rect = cv.resize(image_rect,(100,100))
                    image_num = img_to_array(image_rect)

                    image_num = np.array(image_num).reshape(-1,100,100,1)
                    image_num = image_num.astype('float32')
                    image_num = image_num / 255

                    model = tf.keras.models.load_model('sudoku_model')
                    prediction  = model.predict(image_num)
                    digit = np.argmax(prediction)
                    prob = np.max(prediction)
                    # plt.imshow(image_rect, cmap='gray')
                    # plt.show()
            print('detected:' , digit)
            print('probability' , prob)
            digits_list.append(digit)

    return digits_list

def solve(grid):
    find = find_empty(grid)
    if not find:
        return True
    else:
        row,col = find

    for i in range(1,10):
        if is_valid(grid, i , (row,col)):
            grid[row][col]= i
            if solve(grid):
                return True

            grid[row][col] = 0

    return False


def is_valid(grid,num, coordinate):
    #Check row
    for i in range (len(grid[0])):
        if grid[coordinate[0]][i] == num and coordinate[1] != i:
            return False

    #Check column
    for i in range(len(grid)):
        if grid[i][coordinate[1]] == num and coordinate[0] != i:
            return False


    box_x = coordinate[1]
    box_y = coordinate[0]


    for i in range(box_y*3 , box_y*3 +3):
        for j in range(box_x*3, box_x*3 +3):
            if grid[i][j] == num and (i,j) != coordinate:
                return False

    return True

def find_empty(grid):
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            if grid[i][j] == 0:
                return (i,j)  # row, col

    return None

def save(sudoku2d,sudoku2d_unsolved):
    solved_cell = np.ones((900,900,3))

    for i in range(8):
        solved_cell = cv.line(solved_cell,((i+1)*100,0),((i+1)*100,900),(255,255,255),5)
        solved_cell = cv.line(solved_cell,(0,(i+1)*100),(900,(i+1)*100),(255,255,255),5)

    for i in range(2):
        solved_cell = cv.line(solved_cell,((i+1)*300,0),((i+1)*300,900),(255,255,255),10)
        solved_cell = cv.line(solved_cell,(0,(i+1)*300),(900,(i+1)*300),(255,255,255),10)

font = cv.FONT_HERSHEY_SIMPLEX
fontScale= 2
thickness=4


while True:
    img_resp = requests.get(url)
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
    img = cv.imdecode(img_arr, -1)
    img = imutils.resize(img, width=900, height=900)
    # success, img = frame.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 3)
    canny = cv.Canny(blur, 50, 50)
    copy = img.copy()


    img_contours = img.copy()

    img_contours_bin = get_contours(canny,copy)

    try:
        sudoku = classify(img_contours_bin)
        # print(sudoku)
        sudoku2d = []
        for i in range (0,9):
            sudoku2d.append([cell for cell in sudoku[i*9:(i+1)*9]])
        sudoku2d = np.array(sudoku2d)

        sudoku2d_unsolved = sudoku2d.copy()
        print(sudoku2d_unsolved)

        solve(sudoku2d)
        save_
    except:
        pass

    # get_contours(canny, copy)

    cv.imshow('webcam', copy)
    if cv.waitKey(1) & 0xff == ord('q'):
        break
