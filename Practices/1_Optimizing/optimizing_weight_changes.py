import numpy as np

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

# The data point you will make a prediction for
input_data = np.array([0, 3])

# Sample weights
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]
            }

# The actual target value, used to calculate the error
target_actual = 3

# Make prediction using original weights
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
error_0 = model_output_0 - target_actual

# Create weights that cause the network to make perfect prediction (3): weights_1
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [-1, 1]
            }

# Make prediction using new weights: model_output_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print error_0 and error_1
print(error_0)
print(error_1)

# The goal to compare the errors produced by two different sets of weights to assess the performance of the neural network in predicting the target value.
# Input Data: The variable input_data represents the input features for a neural network. It's a NumPy array containing two elements.
# Sample Weights: Two sets of weights are provided: weights_0 and weights_1. Each set contains weights for two nodes in the first hidden layer (node_0 and node_1), and weights for the output layer.
# Target Actual: The variable target_actual represents the actual target value.
# Prediction Function: The function predict_with_network takes the input data and weights as inputs and predicts the output of the neural network using the ReLU activation function for each node.
# Error Calculation: For each set of weights, the code calculates the model output using the predict_with_network function and then calculates the error by subtracting the actual target value from the predicted output.
# Printing Errors: Finally, the code prints the errors (error_0 and error_1) calculated for the two sets of weights.
