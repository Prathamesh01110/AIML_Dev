import numpy as np

# Define input_data, weights, and target
weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])
target = 0

# Set the learning rate
learning_rate = 0.01

# Calculate the predictions
preds = (weights * input_data).sum()

# Calculate the error
error = preds - target

# Calculate the slope
slope = 2 * input_data * error

# Update the weights
weights_updated = weights - (learning_rate * slope)

# Get updated predictions
preds_updated = (weights_updated * input_data).sum()

# Calculate updated error
error_updated = preds_updated - target

# Print the original error
print("Original Error:", error)

# Print the updated error
print("Updated Error:", error_updated)

# Print weights, target, and input data
print("Weights:", weights)
print("Target:", target)
print("Input Data:", input_data)
