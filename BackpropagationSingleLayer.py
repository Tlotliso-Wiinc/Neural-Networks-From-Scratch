import numpy as np

# Initial inputs
inputs = np.array([1, 2, 3, 4])

# Initial weights and biases
weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

biases = np.array([0.1, 0.2, 0.3])

# Learning rate
learning_rate = 0.001

# ReLU activation function and its derivative
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# Training loop
for iteration in range(200):
    # Forward pass
    z = np.dot(weights, inputs) + biases
    a = relu(z)
    y = np.sum(a)

    # Calculate loss
    loss = y ** 2

    # Backward pass
    # Gradient of the loss with respect to the output y
    dL_dy = 2 * y

    # Gradient of the output y with respect to the ReLU activation a
    dy_da = np.ones_like(a)

    # Gradient of loss with respect to the ReLU activation a
    dL_da = dL_dy * dy_da

    # Gradient of the ReLU activation a with respect to the pre-activation z
    da_dz = relu_derivative(z)

    # Gradient of the loss with respect to the pre-activation z
    dL_dz = dL_da * da_dz

    # Gradient of the pre-activation z with respect to the weights and biases
    dL_dW = np.outer(dL_dz, inputs)
    dL_db = dL_dz

    # Update weights and biases
    weights -= learning_rate * dL_dW
    biases -= learning_rate * dL_db

    # Print the loss for this iteration
    print(f"Iteration {iteration + 1}, Loss: {loss}")


# Final weights and biases
print("Final weights:\n", weights)
print("Final biases:\n", biases)