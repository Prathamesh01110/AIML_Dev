import numpy as np
from sklearn.metrics import mean_squared_error

# Define the relu function
def relu(input):
    return max(0, input)

# Define the prediction function
def predict_with_network(input_data, weights):
    # Calculate node 0 in the first hidden layer
    node_0_input = (input_data * weights['node_0']).sum()
    node_0_output = relu(node_0_input)

    # Calculate node 1 in the first hidden layer
    node_1_input = (input_data * weights['node_1']).sum()
    node_1_output = relu(node_1_input)

    # Put node values into array: hidden_outputs
    hidden_outputs = np.array([node_0_output, node_1_output])
    
    # Calculate model output: model_output
    model_output = (hidden_outputs * weights['output']).sum()
    
    # Return model_output
    return model_output

# Define the provided weights and input data
weights_0 = {'node_0': np.array([2, 1]),
             'node_1': np.array([1, 2]),
             'output': np.array([1, 1])
            }

weights_1 = {'node_0': np.array([2, 1]),
             'node_1': np.array([1., 1.5]),
             'output': np.array([1., 1.5])
            }

input_data = np.array([4, 0])

# Create empty lists to store model outputs
model_output_0 = []
model_output_1 = []

# Loop over input_data
for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

# Calculate the mean squared error for model_output_0: mse_0
mse_0 = mean_squared_error(input_data, model_output_0)

# Calculate the mean squared error for model_output_1: mse_1
mse_1 = mean_squared_error(input_data, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0:", mse_0)
print("Mean squared error with weights_1:", mse_1)
