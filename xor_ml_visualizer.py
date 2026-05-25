import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# 2. XOR Neural Network Class (9 parameters: 4 + 2 + 2 + 1)
class XORNet:
    def __init__(self):
        # Seed for reproducibility
        np.random.seed(0)
        
        # Hidden Layer (2 units)
        # W1: 2x2 = 4 weights
        # b1: 2 = 2 biases
        self.W1 = np.random.uniform(-1, 1, (2, 2))
        self.b1 = np.random.uniform(-1, 1, (1, 2))
        
        # Output Layer (1 unit)
        # W2: 2x1 = 2 weights
        # b2: 1 = 1 bias
        self.W2 = np.random.uniform(-1, 1, (2, 1))
        self.b2 = np.random.uniform(-1, 1, (1, 1))
        
        # Total parameters = 4 + 2 + 2 + 1 = 9
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, output, lr):
        # Output layer error & gradient
        error = y - output
        d_output = error * sigmoid_derivative(output)
        
        # Hidden layer error & gradient
        error_hidden = d_output.dot(self.W2.T)
        d_hidden = error_hidden * sigmoid_derivative(self.a1)
        
        # Save parameter gradients
        dW2 = self.a1.T.dot(d_output)
        db2 = np.sum(d_output, axis=0, keepdims=True)
        dW1 = X.T.dot(d_hidden)
        db1 = np.sum(d_hidden, axis=0, keepdims=True)
        
        # Update parameters
        self.W1 += lr * dW1
        self.b1 += lr * db1
        self.W2 += lr * dW2
        self.b2 += lr * db2

    def get_parameters(self):
        # Flatten all 9 parameters into a dictionary for easy logging
        return {
            'W1_00': self.W1[0, 0], 'W1_01': self.W1[0, 1],
            'W1_10': self.W1[1, 0], 'W1_11': self.W1[1, 1],
            'b1_0': self.b1[0, 0],  'b1_1': self.b1[0, 1],
            'W2_0': self.W2[0, 0],  'W2_1': self.W2[1, 0],
            'b2': self.b2[0, 0]
        }

# 3. Helper to process input byte arrays according to requirements
def process_byte_input(byte_array):
    """
    Takes a byte array (or list of 0s and 1s), splits it in half, and performs XOR.
    Example: 
      Input: [1, 0] -> Halves: [1], [0] -> Output: [1 ^ 0] = [1]
      Input: [1, 0, 0, 1] -> Halves: [1, 0], [0, 1] -> Output: [1^0, 0^1] = [1, 1]
    """
    arr = list(byte_array)
    n = len(arr)
    if n % 2 != 0:
        raise ValueError("Input length must be even to split in half.")
    
    half = n // 2
    first_half = arr[:half]
    second_half = arr[half:]
    
    # Perform element-wise XOR
    xor_result = [a ^ b for a, b in zip(first_half, second_half)]
    return first_half, second_half, xor_result

# 4. Training and Visualization
def train_and_visualize(epochs=10000, lr=0.1):
    # Generated dataset of 20 samples (random binary pairs and their XOR targets)
    np.random.seed(42)  # Seed for dataset reproducibility
    X = np.random.randint(0, 2, size=(20, 2))
    y = np.logical_xor(X[:, 0], X[:, 1]).astype(int).reshape(-1, 1)
    
    model = XORNet()
    
    # History logs for visualization
    history = {
        'loss': [],
        'accuracy': [],
        'W1_00': [], 'W1_01': [], 'W1_10': [], 'W1_11': [],
        'b1_0': [], 'b1_1': [],
        'W2_0': [], 'W2_1': [],
        'b2': []
    }
    
    print("Starting Model Training...")
    print(f"Initial Parameters (Total: 9):")
    for k, v in model.get_parameters().items():
        print(f"  {k}: {v:.4f}")
        
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(X)
        
        # Calculate loss (Mean Squared Error)
        loss = np.mean(np.square(y - predictions))
        
        # Calculate accuracy
        binary_predictions = (predictions >= 0.5).astype(int)
        accuracy = np.mean(binary_predictions == y)
        
        # Log metrics and parameter states
        history['loss'].append(loss)
        history['accuracy'].append(accuracy)
        
        params = model.get_parameters()
        for k, v in params.items():
            history[k].append(v)
            
        # Backward pass & update
        model.backward(X, y, predictions, lr)
        
        # Print progress
        if (epoch + 1) % 1000 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:5d}/{epochs} - Loss: {loss:.6f} - Accuracy: {accuracy*100:5.1f}%")
            
    print("\nTraining Complete!")
    print(f"Final Parameters:")
    for k, v in model.get_parameters().items():
        print(f"  {k}: {v:.4f}")
        
    # Final Model Evaluation
    final_outputs = model.forward(X)
    print("\nModel Prediction Results:")
    for i in range(len(X)):
        print(f"Input Half A: [{X[i][0]}], Half B: [{X[i][1]}] -> Target XOR: {y[i][0]} -> Model Prediction: {final_outputs[i][0]:.4f} (Round: {int(final_outputs[i][0] >= 0.5)})")
        
    # Find the epoch where accuracy first reached 1.0 (converged)
    converged_idx = len(history['accuracy'])
    for idx, acc in enumerate(history['accuracy']):
        if acc == 1.0:
            converged_idx = idx + 1
            break
            
    # Set the plot limit to 350 epochs (or higher if convergence took longer)
    plot_limit = max(350, converged_idx)
    
    # Slice the training history up to the plot limit
    epochs_range = range(1, plot_limit + 1)
    loss_slice = history['loss'][:plot_limit]
    accuracy_slice = history['accuracy'][:plot_limit]
        
    # Plotting results
    plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left subplot: Loss and Accuracy
    color = 'tab:red'
    axes[0].set_xlabel('Epochs', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', color=color, fontsize=12)
    axes[0].plot(epochs_range, loss_slice, color=color, linewidth=2, label='Loss')
    axes[0].tick_params(axis='y', labelcolor=color)
    
    ax2 = axes[0].twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)
    ax2.plot(epochs_range, accuracy_slice, color=color, linewidth=2, linestyle='--', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    axes[0].set_title(f'Training Loss & Accuracy History\n(Plotted: {plot_limit} Epochs | Converged: {converged_idx} Epochs)', fontsize=14, fontweight='bold')
    
    # Right subplot: Trajectory of the 9 parameters
    param_keys = ['W1_00', 'W1_01', 'W1_10', 'W1_11', 'b1_0', 'b1_1', 'W2_0', 'W2_1', 'b2']
    colors = plt.cm.tab10(np.linspace(0, 1, 9))
    
    for key, c in zip(param_keys, colors):
        label_name = f"Hidden W: {key}" if 'W1' in key else (f"Hidden B: {key}" if 'b1' in key else (f"Output W: {key}" if 'W2' in key else "Output B: b2"))
        axes[1].plot(epochs_range, history[key][:plot_limit], label=label_name, color=c, linewidth=1.5)
        
    axes[1].set_xlabel('Epochs', fontsize=12)
    axes[1].set_ylabel('Parameter Value', fontsize=12)
    axes[1].set_title(f'Trajectory of the 9 Model Parameters\n(Plotted: {plot_limit} Epochs | Converged: {converged_idx} Epochs)', fontsize=14, fontweight='bold')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    plt.tight_layout()
    
    # Save the output visualization
    output_filename = "xor_training_process.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization plot as '{output_filename}'")
    return model


# 5. Example usage function to satisfy byte array input requirement
def run_byte_xor_inference_example():
    print("\n" + "="*50)
    print("Demo: Processing 0/1 Byte Array Inputs")
    print("="*50)
    
    # Define some byte arrays (as lists of 0 and 1)
    examples = [
        [0, 1],
        [1, 1],
        [0, 0, 1, 1],
        [1, 0, 1, 0],
        [1, 1, 0, 0, 0, 1, 1, 1]
    ]
    
    for byte_arr in examples:
        half_a, half_b, xor_out = process_byte_input(byte_arr)
        print(f"Input Byte Array: {byte_arr}")
        print(f"  -> Split Half A: {half_a}")
        print(f"  -> Split Half B: {half_b}")
        print(f"  -> Expected XOR Result: {xor_out}")
        print("-" * 30)

# 6. Model saving, loading and parsing helpers
def save_model(model, filename="xor_model.npz"):
    np.savez(filename, W1=model.W1, b1=model.b1, W2=model.W2, b2=model.b2)
    print(f"Model successfully saved to '{filename}'.")

def load_model(filename="xor_model.npz"):
    import os
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Trained model file '{filename}' not found. Please train the model first by running:\n  python xor_ml_visualizer.py learn")
    data = np.load(filename)
    model = XORNet()
    model.W1 = data['W1']
    model.b1 = data['b1']
    model.W2 = data['W2']
    model.b2 = data['b2']
    print(f"Model successfully loaded from '{filename}'.")
    return model

def parse_binary_arg(arg):
    # Remove brackets, commas, quotes, and whitespace to accept formats like "[0, 1]", "0,1", "0 1", or "01"
    clean = arg.replace('[', '').replace(']', '').replace(',', '').replace(' ', '')
    if not clean or not all(c in '01' for c in clean):
        raise ValueError(f"Invalid binary input '{arg}'. Must contain only 0s and 1s.")
    return [int(c) for c in clean]

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("="*60)
        print("XOR Neural Network - Usage Guide:")
        print("  1. Train and save model:")
        print("     python xor_ml_visualizer.py learn")
        print("  2. Test model with 2 concatenated inputs:")
        print("     python xor_ml_visualizer.py test <input1> <input2>")
        print("     (Examples: '01' '11', '[0, 1]' '[1, 0]', or '0' '1')")
        print("="*60)
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    
    if mode == "learn":
        # Run demo first
        run_byte_xor_inference_example()
        # Train and save the model
        model = train_and_visualize(epochs=10000, lr=0.5)
        save_model(model)
        
    elif mode == "test":
        if len(sys.argv) < 4:
            print("Error: 'test' mode requires 2 additional binary input arguments.")
            print("Example:")
            print("  python xor_ml_visualizer.py test 01 11")
            sys.exit(1)
            
        arg1 = sys.argv[2]
        arg2 = sys.argv[3]
        
        try:
            list1 = parse_binary_arg(arg1)
            list2 = parse_binary_arg(arg2)
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
            
        if len(list1) != len(list2):
            print(f"Error: The two inputs must have the same length (got {len(list1)} and {len(list2)}).")
            sys.exit(1)
            
        # Concatenate inputs (붙인뒤)
        combined = list1 + list2
        
        # Split using process_byte_input
        first_half, second_half, xor_expected = process_byte_input(combined)
        
        # Load the saved model
        try:
            model = load_model()
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
            
        # Format input for model forward pass
        X = np.array(list(zip(first_half, second_half)))
        predictions = model.forward(X)
        
        print("\n" + "="*50)
        print("Inference Demo via Command Line Arguments")
        print("="*50)
        print(f"Input 1: {arg1} -> Parse: {list1}")
        print(f"Input 2: {arg2} -> Parse: {list2}")
        print(f"Combined (붙인뒤): {combined}")
        print("-" * 50)
        
        for i in range(len(first_half)):
            pred = predictions[i, 0]
            rounded = int(pred >= 0.5)
            print(f"Bit {i+1}: {first_half[i]} ^ {second_half[i]} | Expected: {xor_expected[i]} | Model prediction: {pred:.4f} (Round: {rounded})")
        print("="*50)
        
    else:
        print(f"Unknown mode: '{mode}'. Use 'learn' or 'test'.")
        sys.exit(1)
