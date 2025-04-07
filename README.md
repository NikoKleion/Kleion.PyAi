# Kleion.PyAi
 An educational deep neural network built from scratch in Python and NumPy. It features dense layers, dropout, batch normalization, learning rate scheduling, and visualization on a custom spiral dataset. A work in progress for learning purposes.


Installation
Prerequisites:
Ensure Python 3.x is installed.

Install Dependencies:
Run:
pip install numpy matplotlib

Clone the Repository:
Clone and navigate into the repository:

bash
Copy
git clone https://github.com/yourusername/advanced_nn-beta.git
cd advanced_nn-beta
Usage
Training Mode
To train the network on a custom spiral dataset, run:
python advanced_nn.py --mode train
This command builds the network, trains it while logging progress and generating visualizations (loss curve, decision boundary, and weight histograms), and saves model checkpoints and a training log CSV.

Testing Mode
To evaluate the saved model, run:
python advanced_nn.py --mode test
This loads the saved parameters and displays the network’s performance on the dataset along with visualizations.

Unit Testing
To run unit tests for core components, run:
python advanced_nn.py --mode unittest

Code Structure
Layers:

Dense: Fully-connected layer with He initialization and L2 regularization.

Activation: Supports ReLU, tanh, and sigmoid functions.

Dropout: Regularizes by randomly dropping neurons during training.

BatchNormalization: Normalizes layer outputs to improve training stability.

Loss Functions:

SoftmaxCrossEntropyLoss: Combines softmax activation with cross-entropy loss.

Optimizer:

SGD: Stochastic Gradient Descent with momentum for parameter updates.

Utilities:

Learning rate scheduling, early stopping, model checkpointing, gradient checking, and CSV logging.

Data Generation:

A spiral dataset is generated to provide a challenging classification task.

Customization
You can adjust training parameters via command-line arguments:

--epochs for the number of training epochs.

--batch_size for the mini-batch size.

--lr for the initial learning rate.

--checkpoint for the model checkpoint file path.

Additionally, the network architecture is defined in the code (under the training section) and can be modified to explore different configurations.

Visualizations
After training, the script produces:

A loss curve showing training loss over epochs.

A decision boundary plot that visualizes the classification regions on the spiral dataset.

Weight distribution histograms for dense layers.

Contributing
Contributions and suggestions are welcome! If you have ideas, improvements, or bug reports, please open an issue or submit a pull request. This project is a learning tool, so feedback is appreciated.
