import numpy as np
from sklearn.metrics import mean_squared_error


def predict_with_network(input_data, weights):

    def relu(x):
        return np.maximum(0, x)

    # Calculate activations for nodes in the first hidden layer
    hidden_0_outputs = np.array([relu(np.dot(input_data, weights[f'node_{i}'])) for i in range(2)])
    print(f"Hidden Layer Output: {hidden_0_outputs}")

    # Calculate activations for nodes in the second hidden layer
    hidden_1_outputs = np.array([relu(np.dot(hidden_0_outputs, weights[f'node_{i}'])) for i in range(2)])
    print(f"Hidden Layer Output: {hidden_1_outputs}")
    # Calculate the model output
    model_output = np.dot(hidden_1_outputs, weights['output'])

    return model_output


# Sample data and weights
input_data = np.array([[0, 3], [0, 3]])  # Two rows for two inputs
target_actuals = np.array([3, 3])  # Two target values
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]}
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [-1, 1]}

# Create model_output_0 
model_output_0 = []
# Create model_output_1
model_output_1 = []

for row in input_data:
    # Append prediction to model_output_0
    model_output_0.append(predict_with_network(row, weights_0))
    
    # Append prediction to model_output_1
    model_output_1.append(predict_with_network(row, weights_1))

mse_0 = mean_squared_error(target_actuals, model_output_0)
mse_1 = mean_squared_error(target_actuals, model_output_1)

# Print mse_0 and mse_1
print("Mean squared error with weights_0: %f" % mse_0)
print("Mean squared error with weights_1: %f" % mse_1)
