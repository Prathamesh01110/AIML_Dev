import numpy as np
import matplotlib.pyplot as plt

# Define input_data, weights, and target
weights = np.array([-0.49929916, 1.00140168, -0.49789747])
input_data = np.array([1, 2, 3])
target = 0

# Define functions to calculate slope and mean squared error
def get_slope(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = preds - target
    slope = 2 * input_data * error
    return slope

def get_mse(input_data, target, weights):
    preds = (weights * input_data).sum()
    error = (preds - target) ** 2
    mse = np.mean(error)
    return mse

# Set the number of updates and create an empty list to store MSE history
n_updates = 20
mse_hist = []

# Iterate over the number of updates
for i in range(n_updates):
    # Calculate the slope
    slope = get_slope(input_data, target, weights)
    
    # Update the weights
    weights = weights - 0.01 * slope
    
    # Calculate MSE with new weights
    mse = get_mse(input_data, target, weights)
    
    # Append the MSE to mse_hist
    mse_hist.append(mse)

# Plot the MSE history
plt.plot(mse_hist)
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.show()

# Print input_data, target, and updated weights
print("Input Data:", input_data)
print("Target:", target)
print("Updated Weights:", weights)
