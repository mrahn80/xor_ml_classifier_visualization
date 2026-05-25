# 🧠 XOR Neural Network Classifier & Visualizer

A premium, lightweight, pure NumPy implementation of a feedforward neural network designed to solve the classic **XOR (Exclusive OR) classification problem** with multi-dimensional split byte array inputs.

This project features a complete end-to-end command-line interface (CLI) to **train** (`learn`), serialize/deserialize trained weights, **test** (`test`) custom bit streams using robust parsers, and automatically generate elegant training performance visualization plots.

---

## 🏗️ Neural Network Architecture

The neural network is a fully connected feedforward network with **9 parameters** designed to map 2-dimensional inputs to a binary output:

*   **Input Layer**: 2 inputs (representing the two halves of a split byte array bit-pair)
*   **Hidden Layer**: 2 neurons with Sigmoid activation ($W_1$: $2 \times 2$ weights, $b_1$: $1 \times 2$ biases)
*   **Output Layer**: 1 neuron with Sigmoid activation ($W_2$: $2 \times 1$ weights, $b_2$: $1 \times 1$ bias)
*   **Total Trainable Parameters**: $4 + 2 + 2 + 1 = 9$ parameters

---

## ⚡ Quick Start & Installation

### 1. Set Up the Virtual Environment

Ensure you have Python 3 installed. Navigate to the project directory and run the following commands to create the environment and install dependencies (`numpy` and `matplotlib`):

```bash
# Create the virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip and install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 How to Run the App

The script runs in two primary modes via command-line arguments: **learn** and **test**.

### 1. Training Mode (`learn`)

Generates a larger dataset of 20 randomized binary training pairs, trains the network until it reaches **100% accuracy** (perfect convergence around **284 epochs**), saves the weights to `xor_model.npz`, and outputs a beautiful training history plot.

```bash
python xor_ml_visualizer.py learn
```

**Key Outputs:**
*   `xor_model.npz`: Compressed archive containing the trained weights and biases (`W1`, `b1`, `W2`, `b2`).
*   `xor_training_process.png`: Plot illustrating the loss curve, accuracy curve, and the trajectories of the 9 network parameters.

### 2. Inference Mode (`test`)

Loads the saved `xor_model.npz` weights, parses two binary input arguments of equal length, concatenates ("붙인뒤") them into a single array, processes them via a split-half XOR operation, and evaluates them through the trained model.

```bash
# Test with simple binary strings
python xor_ml_visualizer.py test "01" "11"

# Test with list/array style formatting
python xor_ml_visualizer.py test "[1, 0, 1]" "[0, 0, 1]"
```

**Example Output:**
```text
Model successfully loaded from 'xor_model.npz'.

==================================================
Inference Demo via Command Line Arguments
==================================================
Input 1: [1, 0, 1] -> Parse: [1, 0, 1]
Input 2: [0, 0, 1] -> Parse: [0, 0, 1]
Combined (붙인뒤): [1, 0, 1, 0, 0, 1]
--------------------------------------------------
Bit 1: 1 ^ 0 | Expected: 1 | Model prediction: 0.9919 (Round: 1)
Bit 2: 0 ^ 0 | Expected: 0 | Model prediction: 0.0065 (Round: 0)
Bit 3: 1 ^ 1 | Expected: 0 | Model prediction: 0.0057 (Round: 0)
==================================================
```

---

## 📊 Training Performance & Parameter Trajectories

The training successfully reaches **100% accuracy** in less than 300 epochs. To visualize the training dynamics, the output plot is automatically trimmed to **350 epochs**, capturing both the active convergence phase and the stable period of the model afterward.

### Output Visualization Plot

![XOR Training Process Plot](./xor_training_process.png)

*   **Left Graph (Loss & Accuracy)**: Tracks the Mean Squared Error (MSE) loss (red) dropping to zero and the classification accuracy (blue) rising smoothly to 100%.
*   **Right Graph (Parameter Trajectory)**: Visualizes the learning journey of each of the 9 weights and biases as they adapt to separate the non-linear XOR space.
