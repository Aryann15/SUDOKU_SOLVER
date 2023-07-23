import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import tensorflow as tf

from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array


myData = 'C://Users//rajly//sudokuSOLVER//data'
categories = ['0','1','2','3','4','5','6','7','8','9']
training_data =[]

def data_augmentation():
    for category in categories:
        path = os.path.join(myData,category)
        class_num = categories.index(category)

        for img_filename in (os.listdir(path)):
            try:
                img_array = cv.imread(os.path.join(path,img_filename),cv.IMREAD_GRAYSCALE)
                canny = cv.Canny(img_array,50,50)
                contours, hierarchy = cv.findContours(canny,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

                for cnt in contours:
                    area = cv.contourArea(cnt)
                    if area > 2:
                        peri = cv.arcLength(cnt,True)
                        approx= cv. approxPolyDP(cnt,0.02*peri,True)
                        x,y,w,h = cv.boundingRect(approx)
                        img_rect = img_array[y:y+h,x:x+w]
                        img_rect = cv.resize(img_rect,(100,100))
                        # plt.imshow(img_rect,cmap= 'gray')
                        # plt.show()

                        kernel = np.ones((3,3),np.uint8)
                        #
                        #
                        for blur_value in range(-30,30):
                            img= cv.GaussianBlur(img_rect,(7,7),blur_value)
                            training_data.append([img, class_num])
                        #     plt.imshow(img,cmap= 'gray')
                        #     plt.show()
                        #
                            img_erosion = cv.erode(img,kernel,iterations =1)
                            img_erosion2 = cv.erode(img,kernel,iterations =2)
                            training_data.append([img_erosion, class_num])
                            training_data.append([img_erosion2, class_num])

                            # plt.imshow(img_erosion2,cmap= 'gray')
                            # plt.show()

                            img_dilation = cv.dilate(img, kernel, iterations=1)
                            img_dilation2 = cv.dilate(img, kernel, iterations=2)
                            training_data.append([img_dilation, class_num])
                            training_data.append([img_dilation2, class_num])

                            # plt.imshow(img_dilation2,cmap= 'gray')
                            # plt.show()

            except Exception as e:
                raise(e)

data_augmentation()

random.seed(3300)
random.shuffle(training_data)

for features,label in training_data[:10]:
    print(label)


IMG_SIZE = 100
X = []
Y = []

for features,label in training_data:
    X.append(features)
    Y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X = X.astype('float32')
X = X /255.0

Y = np.array(Y)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X,Y,epochs=5)
model.save('sudoku_model')