# Kaggle Competition

Competition: https://www.kaggle.com/competitions/digit-recognizer/overview

Submission Accuracy: 87%

Implementation: Vanilla Neural Network

### Competition Description
MNIST ("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

In this competition, your goal is to correctly identify digits from a dataset of tens of thousands of handwritten images. We’ve curated a set of tutorial-style kernels which cover everything from regression to neural networks. We encourage you to experiment with different algorithms to learn first-hand what works well and how techniques compare.

## Implementation of Neural Network

This code uses a simple feed-forward neural network with one hidden layer to classify handwritten digits from the MNIST dataset. The dataset consists of 28x28 pixel images of handwritten digits, and the task is to determine which digit each image represents (a multi-class classification problem with 10 classes, representing the digits 0-9).

## Overview of the Code

The code consists of several sections:

1. **Import Libraries**: The required libraries for this task are imported, which includes numpy, pandas, sklearn.model_selection and matplotlib.

2. **Load Data**: The MNIST data, split into training and test datasets, is loaded from CSV files using pandas.

3. **Activation and Loss Functions**: The code defines several helper functions for the neural network. These include the sigmoid function and its derivative (used as the activation function for the neurons), the softmax function (used to convert the output layer into probabilities), and the categorical cross-entropy function and its derivative (used to calculate the loss and update the weights).

4. **Data Preprocessing**: The labels are one-hot encoded, and the pixel intensities are normalized.

5. **Model Initialization**: The weights and biases for the neural network are randomly initialized.

6. **Training Loop**: The code enters a loop where it trains the neural network for a specified number of epochs. Each epoch consists of a forward pass (calculating the output of the neural network given the current weights), a loss calculation (comparing the output to the true labels), and a backpropagation step (updating the weights to reduce the loss).

7. **Prediction Function**: A function is defined to make predictions on new, unseen data. This function takes in an image, feeds it through the trained network, and returns the digit with the highest probability.

8. **Evaluation**: The code calculates the accuracy of the model on the training data.

## Implementation Details

The neural network uses the sigmoid activation function in the hidden layer and softmax function in the output layer, which makes it a good choice for multi-class classification problems like this one. The loss function is categorical cross-entropy, which is suitable for multi-class classification problems.

The network is trained using gradient descent. In each epoch, the gradient of the loss with respect to the weights is calculated and the weights are updated in the direction that reduces the loss. This process is repeated for a specified number of epochs until the loss stops decreasing significantly.

The network uses a single hidden layer, and the number of nodes in the hidden layer is a hyperparameter that can be adjusted to improve performance.

## Final Remarks

This code provides a simple yet effective way to classify handwritten digits using a neural network. It demonstrates the basic principles of neural network training, including forward propagation, loss calculation, backpropagation, and weight updates.

Please note that this code does not include some techniques that might improve performance, such as validation for hyperparameter tuning, regularization to prevent overfitting, and more sophisticated optimization algorithms like momentum or Adam. However, it can serve as a starting point for learning about and experimenting with neural networks.

### Practice Skills
Computer vision fundamentals including simple neural networks

Classification methods such as SVM and K-nearest neighbors

### Acknowledgements 
More details about the dataset, including algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html. The dataset is made available under a Creative Commons Attribution-Share Alike 3.0 license.