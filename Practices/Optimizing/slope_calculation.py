import numpy as np

# Define input_data, weights, and target
weights = np.array([0, 2, 1])
input_data = np.array([1, 2, 3])
target = 0

# Calculate the predictions: preds
preds = (input_data * weights).sum()

# Calculate the error: error
error = preds - target

# Calculate the slope: slope
slope = error * input_data * 2

# Print the slope
print("Slope:", slope)
print("Weights:", weights)
print("Input data:", input_data)
