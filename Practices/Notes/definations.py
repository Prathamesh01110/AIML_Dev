# Gradient Descent: Gradient descent is an optimization algorithm used to minimize the loss function of a machine learning model. It works by iteratively updating the parameters (weights) of the model in the direction that reduces the loss. At each iteration, the gradient (derivative) of the loss function with respect to each parameter is calculated, and the parameters are adjusted by taking a step proportional to the negative of the gradient.
# Fit Function: In machine learning libraries like TensorFlow or Keras, the fit function is used to train a model on a given dataset. It takes input features (predictors) and corresponding target values as input and adjusts the model parameters (weights) iteratively to minimize the loss function. The fit function typically takes arguments such as the number of epochs (iterations over the dataset), batch size (number of samples processed at once), and validation data, among others.
# Activation Function: An activation function is a mathematical function applied to the output of each neuron (node) in a neural network. It introduces non-linearity into the network, allowing it to learn complex patterns in the data. Common activation functions include ReLU (Rectified Linear Unit), sigmoid, and tanh. ReLU is one of the most widely used activation functions in deep learning due to its simplicity and effectiveness in combating the vanishing gradient problem. It returns 0 for negative inputs and the input value for positive inputs.
# Backpropagation: is the process in which neural networks adjust their weights by propagating the error backward from the output layer to the input layer, using the chain rule of calculus to compute the gradients of the loss function with respect to each parameter. These gradients are then used to update the parameters through optimization algorithms like gradient descent.
# Loss:  Loss = 0.5 * (Prediction - Target)^2
# Gradient (Slope): Slope = -Input * (Target - Prediction)
# Prediction: Prediction = Input * Weight
# New Weight=Old Weight−Learning Rate×Gradient