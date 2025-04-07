#CREATED BY KLEIOOOOOOOOOOOon
#
# Description:
#   This a educational project demonstrates a multi-layer perceptron built entirely
#   from scratch using Python and NumPy. It includes:
#     - Dense layers with He initialization and L2 regularization.
#     - Activations: ReLU, tanh, and sigmoid.
#     - Dropout for regularization.
#     - Batch Normalization to speed up training.
#     - Softmax with cross-entropy loss.
#     - SGD with momentum.
#     - Learning rate scheduling, early stopping, and checkpointing.
#     - Training on a custom spiral dataset.
#     - Visualizations (loss curves, decision boundaries, weight histograms).
#     - Model summary, gradient checking, CSV logging, and a CLI.
#
# Important:
#   * This code is BETA and not finalized; it's meant for learning and experimentation.
#   * Contributions and feedback are welcome!
#
# Usage Tutorial:
#   1. Installation:
#         pip install numpy matplotlib, lol
#
#   2. Running:
#         python advanced_nn.py --mode train
#         python advanced_nn.py --mode test
#         python advanced_nn.py --mode unittest
#
#  3. Command-line args: --epochs, --batch_size, --lr, --checkpoint
#
#   4. Outputs: Training logs (CSV), visual plots, and model checkpoints.
#
# Enjoy exploring and learning!
# ----------------------------------------------------------------

# ----------------- Imports & Global Settings -----------------------
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import argparse
import csv
import unittest
import sys

# Set a fixed random seed for reproducibility
np.random.seed(42)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Base Layer Class
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class Layer:
    def forward(self, input_data, training=True):
        # Each layer must implement its own forward pass
        raise NotImplementedError("Forward method not implemented.")

    def backward(self, grad_output):
        # Each layer must implement its own backward pass
        raise NotImplementedError("Backward method not implemented.")

# ===== Dense (Fully-Connected) Layer =====
class Dense(Layer):
    def __init__(self, input_units, output_units, weight_init='he', l2_reg=0.0):
        # Initialize a dense layer with given number of input/output units.
        self.input_units = input_units
        self.output_units = output_units
        self.l2_reg = l2_reg

        # Weight initialization using He method if specified.
        if weight_init == 'he':
            self.weights = np.random.randn(input_units, output_units) * np.sqrt(2.0 / input_units)
        else:
            self.weights = np.random.randn(input_units, output_units) * 0.01

        self.biases = np.zeros((1, output_units))  # Biases set to zero.
        self.input = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input_data, training=True):
        # Save input for backpropagation.
        self.input = input_data
        output = np.dot(input_data, self.weights) + self.biases
        return output

    def backward(self, grad_output):
        # Backprop: compute gradients w.r.t. input, weights, and biases.
        grad_input = np.dot(grad_output, self.weights.T)
        self.grad_weights = np.dot(self.input.T, grad_output) + self.l2_reg * self.weights
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return grad_input

# ---- Activation Layer (ReLU, tanh, sigmoid) ----
class Activation(Layer):
    def __init__(self, activation):
        # Choose activation function.
        self.activation = activation
        self.input = None

    def forward(self, input_data, training=True):
        self.input = input_data
        if self.activation == 'relu':
            return np.maximum(0, input_data)
        elif self.activation == 'tanh':
            return np.tanh(input_data)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-input_data))
        else:
            raise ValueError("Unsupported activation function")

    def backward(self, grad_output):
        if self.activation == 'relu':
            grad = grad_output.copy()
            grad[self.input <= 0] = 0
            return grad
        elif self.activation == 'tanh':
            return grad_output * (1 - np.tanh(self.input)**2)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-self.input))
            return grad_output * sig * (1 - sig)
        else:
            raise ValueError("Unsupported activation function")

# ~~~ Dropout Layer ~~~
class Dropout(Layer):
    def __init__(self, drop_prob):
        # Initialize dropout with given probability.
        self.drop_prob = drop_prob
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            # Create a dropout mask
            self.mask = (np.random.rand(*input_data.shape) > self.drop_prob) / (1 - self.drop_prob)
            return input_data * self.mask
        else:
            return input_data

    def backward(self, grad_output):
        return grad_output * self.mask

# ---------------- Batch Normalization Layer -----------------
class BatchNormalization(Layer):
    def __init__(self, input_dim, momentum=0.9, epsilon=1e-5):
        # Batch norm layer normalizes the input to speed up training.
        self.gamma = np.ones((1, input_dim))
        self.beta = np.zeros((1, input_dim))
        self.momentum = momentum
        self.epsilon = epsilon
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))
        self.input = None
        self.x_centered = None
        self.std_inv = None

    def forward(self, input_data, training=True):
        self.input = input_data
        if training:
            batch_mean = np.mean(input_data, axis=0, keepdims=True)
            batch_var = np.var(input_data, axis=0, keepdims=True)
            self.x_centered = input_data - batch_mean
            self.std_inv = 1.0 / np.sqrt(batch_var + self.epsilon)
            x_norm = self.x_centered * self.std_inv
            # Update running mean/variance
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            x_norm = (input_data - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
        return self.gamma * x_norm + self.beta

    def backward(self, grad_output):
        N, D = self.input.shape
        x_norm = self.x_centered * self.std_inv
        dgamma = np.sum(grad_output * x_norm, axis=0, keepdims=True)
        dbeta = np.sum(grad_output, axis=0, keepdims=True)
        dx_norm = grad_output * self.gamma
        dvar = np.sum(dx_norm * self.x_centered, axis=0, keepdims=True) * -0.5 * self.std_inv**3
        dmean = np.sum(dx_norm * -self.std_inv, axis=0, keepdims=True) + dvar * np.mean(-2 * self.x_centered, axis=0, keepdims=True)
        dx = dx_norm * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N

        self.grad_gamma = dgamma
        self.grad_beta = dbeta
        return dx

# +++++ Loss Functions +++++
class Loss:
    def forward(self, y_pred, y_true):
        # Compute loss (to be implemented by subclass).
        raise NotImplementedError("Loss forward method not implemented.")

    def backward(self, y_pred, y_true):
        # Compute gradient of loss.
        raise NotImplementedError("Loss backward method not implemented.")

class SoftmaxCrossEntropyLoss(Loss):
    def forward(self, logits, y_true):
        # Compute softmax cross-entropy loss.
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        N = logits.shape[0]
        loss = -np.sum(y_true * np.log(self.probs + 1e-8)) / N
        return loss

    def backward(self, logits, y_true):
        N = logits.shape[0]
        return (self.probs - y_true) / N

# ~~~~~ Optimizer: SGD with Momentum ~~~~~
class SGD:
    def __init__(self, parameters, learning_rate=0.01, momentum=0.9):
        # Initialize the optimizer with parameter dict.
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}
        for key, param in self.parameters.items():
            self.velocities[key] = np.zeros_like(param)

    def step(self):
        # Update parameters using momentum.
        for key in self.parameters:
            if key.endswith("_grad"):
                continue  # Skip gradient keys
            grad_key = key + "_grad"
            if grad_key in self.parameters:
                if key not in self.velocities:
                    self.velocities[key] = np.zeros_like(self.parameters[key])
                self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * self.parameters[grad_key]
                self.parameters[key] += self.velocities[key]

# ----- Learning Rate Scheduler -----
class LearningRateScheduler:
    def __init__(self, initial_lr, decay_factor, step_size):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.step_size = step_size

    def get_lr(self, epoch):
        # Decay learning rate every 'step_size' epochs.
        return self.initial_lr * (self.decay_factor ** (epoch // self.step_size))

# ===== Early Stopping =====
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.stop = False

    def update(self, current_loss):
        # Update early stopping status.
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.stop = True

# ~~~~ Model Checkpointing ~~~~
class ModelCheckpoint:
    def __init__(self, filepath):
        self.filepath = filepath
        self.best_loss = np.inf

    def save(self, model, loss):
        # Save the model if current loss is lower than the best seen.
        if loss < self.best_loss:
            self.best_loss = loss
            self.save_model(model)

    def save_model(self, model):
        parameters = {}
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                parameters[f"W{i}"] = layer.weights
                parameters[f"b{i}"] = layer.biases
            if isinstance(layer, BatchNormalization):
                parameters[f"gamma{i}"] = layer.gamma
                parameters[f"beta{i}"] = layer.beta
        np.savez(self.filepath, **parameters)
        print(f"Model checkpoint saved to {self.filepath}")

# ~~~~ Utility Functions for Model Save/Load ~~~~~
def load_model(model, filepath):
    # Load model parameters from the specified file.
    if os.path.exists(filepath):
        data = np.load(filepath)
        for i, layer in enumerate(model.layers):
            if isinstance(layer, Dense):
                layer.weights = data[f"W{i}"]
                layer.biases = data[f"b{i}"]
            if isinstance(layer, BatchNormalization):
                layer.gamma = data[f"gamma{i}"]
                layer.beta = data[f"beta{i}"]
        print(f"Model loaded from {filepath}")
    else:
        print(f"No checkpoint found at {filepath}")

# +++++ Neural Network Model Class +++++
class NeuralNetworkModel:
    def __init__(self, layers, loss, optimizer):
        # Initialize the full model with architecture, loss, and optimizer.
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X, training=True):
        out = X
        for layer in self.layers:
            # Pass 'training' flag to layers that need it.
            if isinstance(layer, (Dropout, BatchNormalization)):
                out = layer.forward(out, training)
            else:
                out = layer.forward(out)
        return out

    def backward(self, grad):
        # Perform backpropagation through all layers.
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def train_on_batch(self, X_batch, y_batch):
        # Train on one mini-batch.
        logits = self.forward(X_batch, training=True)
        loss_value = self.loss.forward(logits, y_batch)
        grad_loss = self.loss.backward(logits, y_batch)
        self.backward(grad_loss)
        parameters = {}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                parameters[f"W{i}"] = layer.weights
                parameters[f"W{i}_grad"] = layer.grad_weights
                parameters[f"b{i}"] = layer.biases
                parameters[f"b{i}_grad"] = layer.grad_biases
            if isinstance(layer, BatchNormalization):
                parameters[f"gamma{i}"] = layer.gamma
                parameters[f"gamma{i}_grad"] = layer.grad_gamma
                parameters[f"beta{i}"] = layer.beta
                parameters[f"beta{i}_grad"] = layer.grad_beta
        self.optimizer.parameters = parameters
        self.optimizer.step()
        return loss_value

    def predict(self, X):
        # Predict labels for input X.
        logits = self.forward(X, training=False)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def summary(self):
        # Print the network architecture.
        print("Model Summary:")
        for i, layer in enumerate(self.layers):
            layer_type = type(layer).__name__
            if isinstance(layer, Dense):
                print(f"Layer {i}: {layer_type} (Input: {layer.input_units}, Output: {layer.output_units})")
            elif isinstance(layer, BatchNormalization):
                print(f"Layer {i}: {layer_type} (Input Dim: {layer.gamma.shape[1]})")
            elif isinstance(layer, Activation):
                print(f"Layer {i}: {layer_type} (Activation: {layer.activation})")
            elif isinstance(layer, Dropout):
                print(f"Layer {i}: {layer_type} (Drop Prob: {layer.drop_prob})")
            else:
                print(f"Layer {i}: {layer_type}")
        print("-" * 50)

# ~~~~~ Data Generation: Spiral Dataset ~~~~~
def generate_spiral_data(points_per_class=100, num_classes=3):
    # Generate a spiral dataset for classification.
    X = np.zeros((points_per_class * num_classes, 2))
    y = np.zeros(points_per_class * num_classes, dtype='uint8')
    for class_number in range(num_classes):
        ix = range(points_per_class * class_number, points_per_class * (class_number + 1))
        r = np.linspace(0.0, 1, points_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points_per_class) \
            + np.random.randn(points_per_class) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y

# ++++ Utility: One-Hot Encoding ++++
def one_hot_encode(y, num_classes):
    # Convert integer labels to one-hot encoding.
    return np.eye(num_classes)[y.reshape(-1)]

# ===== Visualization: Plot Training Loss =====
def plot_loss(loss_history):
    # Plot training loss curve.
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.legend()
    plt.show()

# --- Visualization: Plot Decision Boundary ---
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid)
    Z = preds.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.5, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.Spectral)
    plt.title("Decision Boundary")
    plt.show()

# ^^^^^ Visualization: Weight Distributions ^^^^^
def plot_weight_distributions(model):
    # Plot weight histograms for each Dense layer.
    num_dense = sum(1 for layer in model.layers if isinstance(layer, Dense))
    plt.figure(figsize=(12, 3 * num_dense))
    plot_index = 1
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            plt.subplot(num_dense, 1, plot_index)
            plt.hist(layer.weights.flatten(), bins=30, alpha=0.7)
            plt.title(f"Weight Distribution of Dense Layer {i}")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")
            plot_index += 1
    plt.tight_layout()
    plt.show()

# ==== Logger: CSV Training Logger ====
class Logger:
    def __init__(self, log_file="training_log.csv"):
        # Initialize CSV logger.
        self.log_file = log_file
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Loss", "Accuracy", "LearningRate", "Time(s)"])

    def log(self, epoch, loss, accuracy, lr, elapsed):
        # Append epoch training info to the CSV.
        with open(self.log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{loss:.6f}", f"{accuracy:.4f}", f"{lr:.6f}", f"{elapsed:.2f}"])

# ~~~~~ Gradient Checking Function ~~~~~
def gradient_checking(model, X_sample, y_sample, epsilon=1e-7):
    # Perform gradient checking to verify backprop.
    logits = model.forward(X_sample, training=True)
    loss_val = model.loss.forward(logits, y_sample)
    grad_loss = model.loss.backward(logits, y_sample)
    model.backward(grad_loss)

    parameters = {}
    gradients = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, Dense):
            parameters[f"W{i}"] = layer.weights
            parameters[f"b{i}"] = layer.biases
            gradients[f"W{i}"] = layer.grad_weights
            gradients[f"b{i}"] = layer.grad_biases
        if isinstance(layer, BatchNormalization):
            parameters[f"gamma{i}"] = layer.gamma
            parameters[f"beta{i}"] = layer.beta
            gradients[f"gamma{i}"] = layer.grad_gamma
            gradients[f"beta{i}"] = layer.grad_beta

    def compute_loss_for_param(param_key, param_matrix):
        original_value = param_matrix.copy()
        param_matrix.flat[0] += epsilon
        loss_plus = model.loss.forward(model.forward(X_sample, training=True), y_sample)
        param_matrix.flat[0] = original_value.flat[0] - epsilon
        loss_minus = model.loss.forward(model.forward(X_sample, training=True), y_sample)
        param_matrix.flat[0] = original_value.flat[0]
        return (loss_plus - loss_minus) / (2 * epsilon)

    for key in parameters:
        param = parameters[key]
        grad_analytic = gradients[key]
        numerical_grad = compute_loss_for_param(key, param)
        analytic_grad = grad_analytic.flat[0]
        diff = np.abs(numerical_grad - analytic_grad)
        print(f"Gradient check for {key}: Numerical = {numerical_grad:.8f}, Analytic = {analytic_grad:.8f}, Diff = {diff:.8f}")

# ++++ Unit Tests for Key Components ++++
class TestNeuralNetworkComponents(unittest.TestCase):
    def setUp(self):
        # Create a simple network with one Dense layer and sigmoid activation.
        self.dense = Dense(3, 2)
        self.activation = Activation('sigmoid')
        self.X = np.array([[0.1, -0.2, 0.3]])

    def test_dense_forward(self):
        out = self.dense.forward(self.X)
        self.assertEqual(out.shape, (1, 2))

    def test_activation_forward(self):
        z = np.array([[0, 1]])
        out = self.activation.forward(z)
        self.assertEqual(out.shape, (1, 2))

    def test_dense_backward(self):
        out = self.dense.forward(self.X)
        grad = np.ones_like(out)
        grad_input = self.dense.backward(grad)
        self.assertEqual(grad_input.shape, self.X.shape)

    def test_loss(self):
        loss_fn = SoftmaxCrossEntropyLoss()
        logits = np.array([[2.0, 1.0, 0.1]])
        y_true = one_hot_encode(np.array([0]), 3)
        loss_val = loss_fn.forward(logits, y_true)
        self.assertGreater(loss_val, 0)

# ~~~~~ Command-Line Interface for Training & Testing ~~~~~
def parse_args():
    # Parse arguments from the command line.
    parser = argparse.ArgumentParser(description="Train or Test the Advanced Neural Network Model (BETA) by Kleion")
    parser.add_argument("--mode", type=str, default="train", help="Mode: train, test, or unittest")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--checkpoint", type=str, default="model_checkpoint.npz", help="Path for model checkpoint")
    return parser.parse_args()

# ===== Main Training Loop =====
def main_training(args):
    # Generate the spiral dataset.
    X, y = generate_spiral_data(points_per_class=100, num_classes=3)
    y_onehot = one_hot_encode(y, 3)
    num_samples = X.shape[0]

    # Build the network architecture.
    layers = []
    layers.append(Dense(2, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 3, l2_reg=1e-4))

    loss_fn = SoftmaxCrossEntropyLoss()
    dummy_parameters = {}  # Initially empty; updated during training.
    optimizer = SGD(dummy_parameters, learning_rate=args.lr, momentum=0.9)
    model = NeuralNetworkModel(layers, loss_fn, optimizer)

    # Print the model summary.
    model.summary()

    lr_scheduler = LearningRateScheduler(initial_lr=args.lr, decay_factor=0.8, step_size=100)
    early_stopper = EarlyStopping(patience=20, min_delta=1e-4)
    checkpoint = ModelCheckpoint(args.checkpoint)
    logger = Logger("training_log.csv")

    epochs = args.epochs
    batch_size = args.batch_size
    loss_history = []

    print("Starting training...\n")
    start_time_total = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        current_lr = lr_scheduler.get_lr(epoch)
        optimizer.learning_rate = current_lr

        permutation = np.random.permutation(num_samples)
        X_shuffled = X[permutation]
        y_shuffled = y_onehot[permutation]

        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            batch_loss = model.train_on_batch(X_batch, y_batch)
            epoch_loss += batch_loss * X_batch.shape[0]

        epoch_loss /= num_samples
        loss_history.append(epoch_loss)

        predictions = model.predict(X)
        accuracy = np.mean(np.argmax(y_onehot, axis=1) == predictions)
        epoch_time = time.time() - epoch_start

        logger.log(epoch+1, epoch_loss, accuracy, current_lr, epoch_time)
        checkpoint.save(model, epoch_loss)

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {accuracy:.4f} - LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")

        early_stopper.update(epoch_loss)
        if early_stopper.stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    total_time = time.time() - start_time_total
    print(f"\nTraining completed in {total_time:.2f} seconds.")

    plot_loss(loss_history)
    plot_decision_boundary(model, X, y)
    plot_weight_distributions(model)

    print("\nPerforming gradient checking on a small batch:")
    X_sample = X[:5]
    y_sample = y_onehot[:5]
    gradient_checking(model, X_sample, y_sample)

    final_model_path = "final_model.npz"
    checkpoint.save_model(model)
    print(f"Final model saved to {final_model_path}")

# ===== Main Testing Function =====
def main_testing(args):
    # Generate dataset identical to training.
    X, y = generate_spiral_data(points_per_class=100, num_classes=3)
    y_onehot = one_hot_encode(y, 3)

    layers = []
    layers.append(Dense(2, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 128, l2_reg=1e-4))
    layers.append(BatchNormalization(128))
    layers.append(Activation('relu'))
    layers.append(Dropout(0.1))
    layers.append(Dense(128, 3, l2_reg=1e-4))

    loss_fn = SoftmaxCrossEntropyLoss()
    dummy_parameters = {}
    optimizer = SGD(dummy_parameters, learning_rate=0.01, momentum=0.9)
    model = NeuralNetworkModel(layers, loss_fn, optimizer)

    load_model(model, args.checkpoint)

    predictions = model.predict(X)
    accuracy = np.mean(np.argmax(y_onehot, axis=1) == predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

    plot_decision_boundary(model, X, y)
    plot_weight_distributions(model)

# ===== Main Entry Point =====
def main():
    args = parse_args()
    if args.mode == "train":
        main_training(args)
    elif args.mode == "test":
        main_testing(args)
    elif args.mode == "unittest":
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuralNetworkComponents)
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        print("Invalid mode. Choose 'train', 'test', or 'unittest'.")

if __name__ == "__main__":
    main()


# ######################## END OF SCRIPT ########################
#
# Additional Notes:
# ---------------
# - This project is a BETA release by me and is intended for learning.
# - The purpose is to demonstrate how to build and train neural networks from scratch.
# - Feel free to modify the architecture, parameters, and settings.
#
# Happy coding and learning!
