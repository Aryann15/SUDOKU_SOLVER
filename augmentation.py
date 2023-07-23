import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

myData = 'C://Users//rajly//sudokuSOLVER//data'
categories = ['0','1','2','3','4','5','6','7','8','9']

def data_augmentation():
    for category in categories:
        path = os.path.join(myData,category)
        class_num = categories.index(category)

        for img_filename in (os.listdir(path)):
            try:
                img_array = cv.imread(os.path.join(path,img_filename))
                plt.imshow(img_array,cmap= 'gray')
                plt.show()

            except Exception as e:
                raise(e)

data_augmentation()