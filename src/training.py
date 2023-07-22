# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

test_data = pd.read_csv('./data/test.csv')
train_data = pd.read_csv('./data/train.csv')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def one_hot_encode(labels, num_classes):
    encoded_labels = np.zeros((labels.shape[0], num_classes))
    for i, label in enumerate(labels):
        encoded_labels[i][label] = 1
    return encoded_labels

# Update the loss calculation
def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-10
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred))

#size
input_size = 28 * 28
hidden_size = 600
output_size = 10

#instantiate values
num_classes = 10
labels = train_data['label'].to_numpy()
images = train_data.drop(['label'], axis=1).to_numpy()

# Assuming you have X (features) and y (labels) for your dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_labels = one_hot_encode(train_labels, 10)
test_labels = one_hot_encode(test_labels, 10)

#instantiate weights
w_i_h = np.random.uniform(-0.5, 0.5, (input_size, hidden_size)) 
w_h_o = np.random.uniform(-0.5, 0.5, (hidden_size, output_size))

b_i_h = np.zeros(hidden_size)
b_h_o = np.zeros(output_size)

# Training parameters
learning_rate = 1
epochs = 1000

# Training loop
for epoch in range(epochs):
    # Forward pass
    h_pre = np.dot(train_images, w_i_h) + b_i_h
    h = sigmoid(h_pre)
    
    o_pre = np.dot(h, w_h_o) + b_h_o
    o = sigmoid(o_pre)

    # Calculate the loss
    
    #sigmoid loss
    loss = np.mean((o - train_labels) ** 2)

    #categorical_cross_entropy
    #loss = categorical_cross_entropy(train_labels, o)

    # Backpropagation
    output_error = (o - train_labels) * sigmoid_derivative(o)
    hidden_error = np.dot(output_error, w_h_o.T) * sigmoid_derivative(h)

    # Update weights and biases
    w_h_o -= learning_rate * np.dot(h.T, output_error) / len(train_images)
    b_h_o -= learning_rate * np.sum(output_error, axis=0) / len(train_images)
    w_i_h -= learning_rate * np.dot(train_images.T, hidden_error) / len(train_images)
    b_i_h -= learning_rate * np.sum(hidden_error, axis=0) / len(train_images)

    # Print progress
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

# Prediction function
def predict(image):
    h_pre = np.dot(image, w_i_h) + b_i_h
    h = sigmoid(h_pre)
    o_pre = np.dot(h, w_h_o) + b_h_o
    o = sigmoid(o_pre)
    return o.argmax()

# Evaluation on training data
correct = 0
total = len(train_images)
for i in range(total):
    prediction = predict(train_images[i].reshape(1, -1))
    if prediction == train_labels[i].argmax():
        correct += 1

accuracy = correct / total
print(f"Training accuracy: {accuracy * 100:.2f}%")