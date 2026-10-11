# This is the neural network forward propagation function with two hidden layers.
# This is for the binary classification problem so that is why the output layer has the sigmoid activation function.
import numpy as np

def relu(z):
    output = np.maximum(0, z)
    return output
    
def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def sigmoid(z):
    output = 1 / (1 + np.exp(-z))
    return output

def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    # TODO: Complpete the function
    # Your code here
    # input layer to first hidden layer
    y1 = np.dot(W1,X)+b1
    o1 = relu(y1)
    # first hidden layer to second hidden layer
    y2 = np.dot(W2,o1)+b2
    o2 = relu(y2)
    # second hidden layer to output layer
    y3 = np.dot(W3,o2)+b3
    output = sigmoid(y3)    
    
    return output