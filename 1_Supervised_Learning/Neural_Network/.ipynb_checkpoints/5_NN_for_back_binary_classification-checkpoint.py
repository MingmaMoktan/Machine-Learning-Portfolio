import numpy as np

np.random.seed(42) 

# Input layer
x = np.random.rand(3,1)

# First hidden layer weight and bias
w = np.random.rand(4,3) 
b = np.random.rand(4,1) 

# Weight and bias for the second hidden layer
w1 = np.random.rand(3,4) 
b1 = np.random.rand(3,1) 

# Weight and bias for the output layer
w2 = np.random.rand(1,3)  
b2 = np.random.rand(1,1) 

# Activation Function & its Derivative
def relu(a):
    return np.maximum(0,a)

def relu_derivative(a):
    return np.where(a > 0, 1, 0)

def sigmoid(a):
    return 1/(1+np.exp(-a))

def sigmoid_derivative(a):
    s = sigmoid(a)
    return s * (1 - s)


# --- Forward Propagation ---
# input ----> wx+b ----> relu ----> wx+b ----> sigmoid ---->
r1 = np.dot(w,x)+b
o1 = relu(r1)

r2 = np.dot(w1,o1)+b1
o2 = relu(r2)

r3 = np.dot(w2,o2)+b2
o3 = sigmoid(r3)

print("Prediction (o2):", o2)


# --- Backward Propagation ---
# error ----> delta (error*activation_derivative) ----> weight update
target = 1
e = o3 - target # Error (prediction - target)

# 1. Output Layer Gradients
d3 = e * sigmoid_derivative(r3) # Delta for output layer
d_w2 = np.dot(d3, o2.T)
d_b2 = d3

# 2. Second Hidden Layer Gradients
d2 = np.dot(w2.T, d3) * relu_derivative(r2) # Delta for second hidden layer
d_w1 = np.dot(d2, o1.T)
d_b1 = d2

# 3. First Hidden Layer Gradients
d1 = np.dot(w1.T, d2) * relu_derivative(r1) 
d_w = np.dot(d1, x.T)                 
d_b = d1              


# --- Parameter Updates ---
learning_rate = 0.01

w2 = w2 - learning_rate * d_w2
b2 = b2 - learning_rate * d_b2

w1 = w1 - learning_rate * d_w1
b1 = b1 - learning_rate * d_b1

w  = w  - learning_rate * d_w
b  = b  - learning_rate * d_b