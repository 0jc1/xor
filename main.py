import numpy as np
from itertools import product
import sklearn.neural_network
import time

inputs = 4

def generate_xor_data(num_inputs):
    # Generate all possible binary input combinations
    inputs = list(product([0, 1], repeat=num_inputs))
    X = np.array(inputs)
    
    # Calculate XOR output for each input combination
    y = np.zeros(len(X))
    for i, x in enumerate(X):
        result = x[0]
        for j in range(1, len(x)):
            result = result ^ x[j]  # XOR operation
        y[i] = result
    
    return X, y

# Define the XOR input and output data
X, y = generate_xor_data(inputs)

# Build the neural network model
model = sklearn.neural_network.MLPClassifier(
                activation='relu',
                max_iter=1000,
                hidden_layer_sizes=(100,),
                solver='adam',)

start_time = time.time()

model.fit(X, y)

# Get predictions
predictions = model.predict(X)

training_time = time.time() - start_time

print(f"Training completed in {training_time:.4f} seconds")

# Calculate accuracy
accuracy = np.mean(predictions == y) * 100  # as percentage

print('Model score (accuracy):', model.score(X, y))
print('Predictions:', predictions)
print('True labels:', y)
print(f'Accuracy: {accuracy:.2f}%')
