from sklearn.model_selection import train_test_split
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import csv 

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

#instantiate values
num_classes = 10
labels = train_data['label'].to_numpy()
images = train_data.drop(['label'], axis=1).to_numpy()

# Assuming you have X (features) and y (labels) for your dataset
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
train_labels = one_hot_encode(train_labels, 10)
test_labels = one_hot_encode(test_labels, 10)

# Load the weights from the .npy files
w_i_h = np.load('./output/weights_w_i_h.npy')
w_h_o = np.load('./output/weights_w_h_o.npy')
b_i_h = np.load('./output/weights_b_i_h.npy')
b_h_o = np.load('./output/weights_b_h_o.npy')

# Prediction function
def predict(image):
    h_pre = np.dot(image, w_i_h) + b_i_h
    h = sigmoid(h_pre)
    o_pre = np.dot(h, w_h_o) + b_h_o
    o = sigmoid(o_pre)
    return o.argmax()

# Evaluation on training data
correct = 0
total = len(test_images)
for i in range(total):
    prediction = predict(test_images[i].reshape(1, -1))
    if prediction == test_labels[i].argmax():
        correct += 1

accuracy = correct / total
print(f"Training accuracy: {accuracy * 100:.2f}%")

# Prediction
f = open('./output/submission_output.csv', 'w')

# create the csv writer
writer = csv.writer(f)

test_data = test_data.to_numpy()
writer.writerow(['ImageId','Label'])
# write a row to the csv file
index = 1
for row in test_data:
    prediction = predict(row.reshape(1, -1))
    writer.writerow([index,prediction])
    index = index + 1

# close the file
f.close()