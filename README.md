# SUDOKU SOLVER
## Overview

Sudoku Solver is an AI-powered project that uses computer vision techniques, deep learning, and backtracking algorithms to automatically solve Sudoku puzzles. The solver can take an image of a Sudoku puzzle as input, extract individual digits, recognize them using a deep learning model, and then apply a backtracking algorithm to find the correct solution.

## Features

- Image Processing: The solver uses OpenCV for image processing, including contour detection and perspective transformation, to extract and preprocess individual digits from the Sudoku puzzle image.

- Deep Learning: Sudoku digits are classified using a trained neural network model, which accurately recognizes handwritten digits.

- Backtracking Algorithm: The solver employs a backtracking algorithm to solve the Sudoku puzzle, checking for valid digits in each cell and backtracking when necessary to find the correct solution.

## Dependencies

The Sudoku Solver relies on the following libraries:

- OpenCV: For image processing and contour detection.
- TensorFlow: For deep learning and neural network model inference.
- Keras: For building and training the digit recognition model.
- request: Used to fetch the image from the smartphone's camera server (IP Webcam) using the provided URL.(no need if using laptop' webcam)
- imutils: Used for resizing the image captured from the smartphone's camera.(no need if using laptop' webcam)
  
Make sure you have these libraries installed before running the solver.

**NOTE: If (like me) you do not have access to a good webcam on your laptop or an external webcam, you can use your smartphone's camera to capture the Sudoku puzzle image and then utilize the application ("IP Webcam") to stream the camera feed to the laptop.**

 The solver.py script can be modified to accommodate both scenarios:
 
 ### For smartphone camera usage:
 
import cv2 as cv <br>
import numpy as np <br>
import tensorflow as tf <br>
import requests <br>
import utils <br>
**(URL for the IP Webcam server stream)** <br>
url = "http://your_phone_ip_address:8080/shot.jpg" <br>
**Rest of the code remains the same as in the original solver.py script**
...

 ### For Laptop Webcam usage:
 
import cv2 as cv <br>
import numpy as np <br>
import tensorflow as tf <br>
**OpenCV function to access the laptop's webcam**  <br>
cap = cv.VideoCapture(0) <br>
**Rest of the code remains the same as in the original solver.py script...**


## Contributing

Contributions to Sudoku Solver are welcome! If you find any issues, have ideas for improvement, or want to add new features, feel free to open an issue or submit a pull request.

## Contact

For any inquiries or questions regarding the Sudoku Solver, feel free to contact me at rajlysm15@gmail.com

Happy Sudoku Solving!


