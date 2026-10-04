import numpy as np

# 1. Your original predicted probabilities matrix (5 samples x 3 classes)
# y_pred = np.array([
#     [0.6,   0.15,  0.25],
#     [0.1,   0.2,   0.7 ],
#     [0.2,   0.35,  0.45],
#     [0.1,   0.5,   0.4 ],
#     [0.5,   0.2,   0.3 ]
# ])
y_pred = np.array([
    [0.8,   0.15,  0.05],
    [0.1,   0.15,  0.75],
    [0.1,   0.15,  0.75],
    [0.1,   0.7,   0.2 ],
    [0.6,   0.2,   0.2 ]
])

# 2. Your true labels as integer indices (for the 5 samples)
y_true = np.array([0, 2, 2, 1, 0])

def calculate_cross_entropy(y_pred, y_true):
    # Small epsilon to prevent log(0) errors if a probability is exactly 0
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Advanced indexing to grab the probability of the true class for each sample
    correct_class_probs = y_pred_clipped[np.arange(len(y_true)), y_true]
    
    # Calculate the mean negative log-likelihood (Cross-Entropy Loss)
    loss = -np.mean(np.log(correct_class_probs))
    
    return loss

# Calculate and print the result
loss_value = calculate_cross_entropy(y_pred, y_true)
print(f"Cross-entropy loss: {loss_value:.3f}")