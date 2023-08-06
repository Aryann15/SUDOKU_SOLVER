# SUDOKU_SOLVER
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

Make sure you have these libraries installed before running the solver.

## Contributing

Contributions to Sudoku Solver are welcome! If you find any issues, have ideas for improvement, or want to add new features, feel free to open an issue or submit a pull request.

## Contact

For any inquiries or questions regarding the Sudoku Solver, feel free to contact me at rajlysm15@gmail.com

Happy Sudoku Solving!


