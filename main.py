import numpy as np
from sklearn.neural_network import MLPClassifier
from itertools import product

def generate_xor_data(num_inputs):
    """Generate training data for XOR with specified number of inputs."""
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

def create_and_train_model(X, y):
    """Create and train neural network for XOR problem."""
    # Create MLPClassifier with appropriate architecture
    # - One hidden layer with 2^(num_inputs) neurons
    # - ReLU activation for better training
    # - Adam optimizer for efficient training
    model = MLPClassifier(
        hidden_layer_sizes=(2**len(X[0]),),
        activation='relu',
        solver='adam',
        random_state=42,
        max_iter=1000
    )
    
    # Train the model
    model.fit(X, y)
    return model

def test_model(model, X, y):
    """Test model predictions against expected outputs."""
    predictions = model.predict(X)
    accuracy = model.score(X, y)
    return predictions, accuracy

def run_integration_tests(num_inputs=3):
    """Run integration tests for the XOR neural network."""
    # Test data generation
    X, y = generate_xor_data(num_inputs)
    assert len(X) == 2**num_inputs, f"Expected {2**num_inputs} input combinations"
    
    # Test model creation and training
    model = create_and_train_model(X, y)
    assert model.n_layers_ == 3, "Model should have 3 layers (input, hidden, output)"
    
    # Test predictions
    predictions, accuracy = test_model(model, X, y)
    assert accuracy > 0.99, f"Model accuracy should be >99%, got {accuracy*100:.2f}%"
    
    # Test specific XOR cases
    test_cases = list(product([0, 1], repeat=num_inputs))
    for inputs in test_cases:
        prediction = model.predict([inputs])[0]
        expected = 0
        for bit in inputs:
            expected ^= bit
        assert abs(prediction - expected) < 0.01, f"Failed for inputs {inputs}"
    
    print("All integration tests passed!")
    return True

def main(num_inputs=3):
    """Main function to demonstrate XOR neural network."""
    # Generate training data
    X, y = generate_xor_data(num_inputs)
    
    # Create and train model
    model = create_and_train_model(X, y)
    
    # Test model
    predictions, accuracy = test_model(model, X, y)
    
    # Print results
    print(f"\nResults for {num_inputs}-input XOR:")
    print("Input combinations -> Predicted (Expected)")
    print("-" * 40)
    for inputs, pred, exp in zip(X, predictions, y):
        inputs_str = ', '.join(map(str, inputs))
        print(f"[{inputs_str}] -> {pred:.0f} ({exp:.0f})")
    print(f"\nModel accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    binary_inputs = 3
    
    # Run integration tests first
    run_integration_tests(binary_inputs)
    
    # Run main demonstration
    main(binary_inputs)
