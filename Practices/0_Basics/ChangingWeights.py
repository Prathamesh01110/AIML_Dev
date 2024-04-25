#Our target prediction is 3 so to achive it we will change the weights of it and that will reduces the losses currently at initally weights it is giving us prediction as 27 which we will change it to 3
import numpy as np

def predict_with_network(input_data, weights):
 
  def relu(x):

    return max(0, x)

  # Calculate activations for nodes in the first hidden layer
  hidden_0_outputs = np.array([relu(np.dot(input_data, weights[f'node_{i}'])) for i in range(2)])
  print(f"Activations for first hidden layer: {hidden_0_outputs}")

  # Calculate activations for nodes in the second hidden layer
  hidden_1_outputs = np.array([relu(np.dot(hidden_0_outputs, weights[f'node_{i}'])) for i in range(2)])
  print(f"Activations for second hidden layer: {hidden_1_outputs}")

  # Calculate the model output
  model_output = np.dot(hidden_1_outputs, weights['output'])
  print(f"Output/Prediction is: {model_output}")

  return model_output


# Sample data and weights
input_data = np.array([0, 3])
weights_0 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [1, 1]}
weights_1 = {'node_0': [2, 1],
             'node_1': [1, 2],
             'output': [-1, 1]}

# Make prediction using weights_0
model_output_0 = predict_with_network(input_data, weights_0)

# Calculate error: error_0
target_actual = 3
error_0 = model_output_0 - target_actual

# Make prediction using weights_1
model_output_1 = predict_with_network(input_data, weights_1)

# Calculate error: error_1
error_1 = model_output_1 - target_actual

# Print errors
print(f"Error with weights_0: {error_0}")
print(f"Error with weights_1: {error_1}")
