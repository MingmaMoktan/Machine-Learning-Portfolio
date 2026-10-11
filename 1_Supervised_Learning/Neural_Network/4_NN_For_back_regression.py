import numpy as np

np.random.seed(42) # This is for the reproducible numbers

# Input layer
x = np.random.rand(3,1) # This is input

# First Hidden layer weight and bias
w = np.random.rand(4,3) # Weight
b = np.random.rand(4,1) # Bias

# Weight and bias for the output layer
w1 = np.random.rand(1,4) # Weight
b1 = np.random.rand(1,1) # Bias

# Activation Function & its Derivative
def relu(a):
    return np.maximum(0, a)

def relu_derivative(a):
    # Derivative of ReLU is 1 if a > 0, else 0
    return np.where(a > 0, 1, 0)

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

# --- Forward Propagation ---
# input ----> wx+b ----> relu ----> wx+b ----> relu ----> output
r1 = np.dot(w, x) + b
o1 = relu(r1)

r2 = np.dot(w1, o1) + b1
o2 = relu(r2)

print("Prediction (o2):", o2)

# --- Backward Propagation ---
# output ----> error ----> delta (output_error*relu_derivative) ----> weight update
# Assuming target y = 1
target = 1
output_error = o2 - target # (prediction - target)

# 1. Output layer delta (incorporating the activation derivative) using chain rule
delta2 = output_error * relu_derivative(r2)

# 2. Gradients for output layer (w1 and b1)
d_w1 = np.dot(delta2, o1.T)
d_b1 = delta2

# 3. Hidden layer error / delta chain rule: delta1 = (w1.T * delta2) * relu_derivative(r1)
delta1 = np.dot(w1.T, delta2) * relu_derivative(r1)

# 4. Gradients for hidden layer (w and b)
d_w = np.dot(delta1, x.T)
d_b = delta1

# --- Weight Update Step (Gradient Descent) ---
learning_rate = 0.01

w1 = w1 - learning_rate * d_w1
b1 = b1 - learning_rate * d_b1
w  = w  - learning_rate * d_w
b  = b  - learning_rate * d_b

print("Updated Output Weight (w1):", w1)