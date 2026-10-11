import numpy as np

np.random.seed(42) # This is for the reporducable numbers

# Input layer
x = np.random.rand(3,1) # This is input

# First Hidden layer weight and bias
w = np.random.rand(4,3) # Weight
b = np.random.rand(4,1) # Bias

# Weight and bias for the output layer
w1 = np.random.rand(1,4) # Weight
b1 = np.random.rand(1,1) # Bias

# Activation Function
def relu(a):
    o = np.maximum(0,a)
    return o

def sigmoid(a):
    o = 1/(1+np.exp(-a))

r1 = np.dot(w,x)+b
o1 = relu(r1)

r2 = np.dot(w1,o1)+b1
o2 = relu(r2)