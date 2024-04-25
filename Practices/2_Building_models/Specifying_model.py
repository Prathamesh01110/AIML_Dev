import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# Define predictors (assuming it's a NumPy array with some data)
predictors = np.random.rand(100, 10)  # Example data with 100 samples and 10 features

# Save the number of columns in predictors: n_cols
n_cols = predictors.shape[1]

# Set up the model: model
model = Sequential()

# Add the first layer
model.add(Dense(50, activation='relu', input_shape=(n_cols,)))

# Add the second layer
model.add(Dense(32, activation='relu'))

# Add the output layer
model.add(Dense(1))

# Print model summary
model.summary()
