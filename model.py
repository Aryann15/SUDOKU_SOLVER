import numpy as np
import cv2 as cv
import os
import random
import tensorflow as tf
from sklearn.model_selection import KFold
from statistics import *
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

path= 'data'
training_data = []

myList = os.listdir(path)
print(myList)

class_numbers = len(myList)
for x in range(0,class_numbers):
    pic_list = os.listdir(path + '/' + str(x))
    for y in pic_list:
        current_image = cv.imread(path+'/'+str(x)+ '/' +y)
        current_image = cv.resize(current_image,(100,100))
        gray = cv.cvtColor(current_image,cv.COLOR_BGR2GRAY)
        training_data.append([gray,x])
    print(x)

training_data = np.array(training_data, dtype=object)

print(len(training_data))

random.seed(10160)
random.shuffle(training_data)
for features, labels in training_data[:10]:
    print(labels)

X=[]
Y=[]

for features,labels in training_data:
    X.append(features)
    Y.append(labels)

X=np.array(X)
X=X.astype('float32')
X=X/255
print(len(X))

Y=np.array(Y)


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