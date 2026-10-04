import numpy as np

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - np.max(z, axis=0, keepdims=True))
    return e / np.sum(e, axis=0, keepdims=True)

def sigmoid(z):
    output = 1 / (1 + np.exp(-z))
    return output
    
def forward_propagation(X, W1, b1, W2, b2, W3, b3):
    # TODO: Complpete the function
    # Your code here
    # Input layer to first hidden layer
    y1 = np.dot(W1,X)+b1
    o1 = relu(y1)
    # First hidden layer to second hidden layer
    y2 = np.dot(W2,o1)+b2
    o2 = relu(y2)
    # Second hidden layer to output layer
    y3 = np.dot(W3,o2)+b3
    output = softmax(y3)
    
    return output

# Example input (5 features)
X = np.array([[0.5], [-1.0], [2.0], [0.0], [1.5]])

# TODO: Define weights/biases with correct shapes (fill the blanks)
W1 = np.random.rand(4,5 )
b1 = np.random.rand(4,1)

W2 = np.random.rand(6,4)
b2 = np.random.rand(6,1)

W3 = np.random.rand(3,6)
b3 = np.random.rand(3,1)

print(forward_propagation(X, W1, b1, W2, b2, W3, b3))