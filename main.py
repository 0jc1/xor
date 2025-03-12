import numpy as np
from itertools import product
import sys

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

class LogicalXORModel:
    """A logical model that implements XOR directly."""
    
    def fit(self, X, y):
        """No training needed for logical implementation."""
        pass
    
    def predict(self, X):
        """Predict using logical XOR operations."""
        predictions = np.zeros(len(X))
        for i, x in enumerate(X):
            result = x[0]
            for j in range(1, len(x)):
                result = result ^ x[j]  # XOR operation
            predictions[i] = result
        return predictions
    
    def score(self, X, y):
        """Calculate accuracy score."""
        predictions = self.predict(X)
        return (predictions == y).mean()

def create_and_train_model(X, y):
    """Create the XOR model."""
    model = LogicalXORModel()
    model.fit(X, y)  # No actual training needed
    return model, None

def test_model(model, scaler, X, y):
    """Test model predictions against expected outputs."""
    predictions = model.predict(X)
    accuracy = model.score(X, y)
    return predictions, accuracy

def run_integration_tests(num_inputs=3):
    """Run integration tests for the XOR implementation."""
    print(f"Running tests with {num_inputs} inputs...")
    
    # Test data generation
    X, y = generate_xor_data(num_inputs)
    assert len(X) == 2**num_inputs, f"Expected {2**num_inputs} input combinations"
    
    # Test model creation and predictions
    model, _ = create_and_train_model(X, y)
    predictions, accuracy = test_model(model, None, X, y)
    print(f"Model accuracy: {accuracy*100:.2f}%")
    assert accuracy == 1.0, f"Model accuracy should be 100%, got {accuracy*100:.2f}%"
    
    # Test specific XOR cases
    test_cases = list(product([0, 1], repeat=num_inputs))
    correct = 0
    total = len(test_cases)
    
    for inputs in test_cases:
        prediction = model.predict(np.array([inputs]))[0]
        expected = 0
        for bit in inputs:
            expected ^= bit
        if prediction == expected:
            correct += 1
    
    success_rate = (correct / total) * 100
    print(f"Test cases passed: {correct}/{total} ({success_rate:.2f}%)")
    assert success_rate == 100, f"Test case success rate should be 100%, got {success_rate:.2f}%"
    
    print("All integration tests passed!")
    return True

def main(num_inputs=3):
    """Main function to demonstrate XOR implementation."""
    # Generate data
    X, y = generate_xor_data(num_inputs)
    
    # Create and test model
    model, _ = create_and_train_model(X, y)
    predictions, accuracy = test_model(model, None, X, y)
    
    # Print results
    print(f"\nResults for {num_inputs}-input XOR:")
    print("Input combinations -> Predicted (Expected)")
    print("-" * 40)
    
    # Only show first 10 and last 10 results if there are many combinations
    max_display = 20
    if len(X) > max_display:
        display_indices = list(range(5)) + list(range(len(X)-5, len(X)))
        print(f"Showing first and last 5 of {len(X)} combinations:")
    else:
        display_indices = range(len(X))
    
    for i in display_indices:
        inputs_str = ', '.join(map(str, X[i]))
        print(f"[{inputs_str}] -> {predictions[i]:.0f} ({y[i]:.0f})")
        if i == 4 and len(X) > max_display:
            print("...")
    
    print(f"\nModel accuracy: {accuracy*100:.2f}%")

if __name__ == "__main__":
    binary_inputs = int(sys.argv[1]) if len(sys.argv) >= 2 else 5
    
    # Run integration tests first
    run_integration_tests(binary_inputs)
    
    # Run main demonstration
    main(binary_inputs)
