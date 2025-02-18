# DeepVision: MNIST Classification with OOP
DeepVision: MNIST Classification using OOP  This repository implements three MNIST classifiers—Random Forest, FFNN, and CNN—using a unified object-oriented interface for training, preprocessing, prediction, and validation. It includes a demo Jupyter Notebook, documentation, and necessary configuration files.

1. **Random Forest** (via scikit-learn)  
2. **Feed-Forward Neural Network (FFNN)** (via TensorFlow/Keras)  
3. **Convolutional Neural Network (CNN)** (via TensorFlow/Keras)

All models share a common interface and are accessible through a unifying `MnistClassifier` wrapper.

---

## Table of Contents

1. [Project Structure](#project-structure)  
2. [Requirements](#requirements)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Demo Notebook](#demo-notebook)  
6. [Edge Cases](#edge-cases)  

---

## Project Structure
```
DeepVision/
├── src/
│   ├── __init__.py
│   ├── mnist_classifier_interface.py  # Abstract interface with train, predict, etc.
│   ├── random_forest_classifier.py    # RandomForestMnistClassifier
│   ├── ffnn_classifier.py             # FfnnMnistClassifier
│   ├── cnn_classifier.py              # CnnMnistClassifier
│   └── mnist_classifier.py            # MnistClassifier wrapper to choose among rf, nn, cnn
├── demo_notebook/
│   └── DeepVision_MNIST_Classification.ipynb  # Demonstration Notebook
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

### Main Components

- **`mnist_classifier_interface.py`**: Defines the `MnistClassifierInterface`, an abstract base class specifying `preprocess`, `train`, `predict`, and `validate`.
- **`random_forest_classifier.py`**: Implements a Random Forest model using scikit-learn, conforming to the interface.
- **`ffnn_classifier.py`**: Implements a feed-forward neural network using TensorFlow/Keras, conforming to the interface.
- **`cnn_classifier.py`**: Implements a convolutional neural network using TensorFlow/Keras, conforming to the interface.
- **`mnist_classifier.py`**: A unifying wrapper that instantiates one of the above models based on the `algorithm` parameter (`rf`, `nn`, or `cnn`).
- **`DeepVision_MNIST_Classification.ipynb`**: Demonstrates how to load MNIST, train each model, evaluate performance, and visualize predictions.

---

## Requirements

The essential libraries are listed in `requirements.txt`. A minimal set typically includes:

- `numpy`
- `scikit-learn`
- `tensorflow`
- `keras`
- `matplotlib`
- `notebook`

---

## Installation

1. **Clone the repository**:
```bash
   git clone https://github.com/AnDeresh/DeepVision.git
   cd DeepVision
```

2. **Create and activate a virtual environment** (recommended):
- On Windows (Command Prompt):
``` bash
    python -m venv venv
    .\venv\Scripts\activate
```

- On Linux/Mac:
``` bash
    python3 -m venv venv
    source venv/bin/activate
```

3. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

## Usage

1. Using the Models in Your Own Scripts
You can import and use the classes directly in Python scripts:
```bash
    from src.mnist_classifier import MnistClassifier

    # Example for CNN
    classifier = MnistClassifier(algorithm='cnn', input_shape=(28, 28, 1), num_classes=10)
    classifier.train(X_train, y_train, X_val, y_val, epochs=5, batch_size=64)
    predictions = classifier.predict(X_test)
```

2. Notebook Demo
- Open the demo notebook:
```bash
jupyter notebook demo_notebook/DeepVision_MNIST_Classification.ipynb
```
- Run the cells in the notebook to:
1. Load MNIST data
2. Train and validate each model (Random Forest, FFNN, CNN)
3. Evaluate performance on the test set
4. Visualize sample predictions

## Demo Notebook

The `DeepVision_MNIST_Classification.ipynb` notebook walks you through:

1. Data Loading: Fetching MNIST via `tensorflow.keras.datasets.mnist`.
2. Preprocessing: Normalizing pixel values, reshaping images if necessary.
3. Training: Demonstrating how to train each model (Random Forest, FFNN, CNN).
4. Validation & Testing: Calculating accuracy scores on validation and test sets.
5. Visualization: Displaying sample images and predicted labels.

## Edge Cases

1. **Incorrect Input Shapes**
- FFNN expects `(28, 28)` by default.
- CNN expects `(28, 28, 1)`.
- Random Forest automatically flattens images but assumes shape `(n_samples, 28, 28)`.

2. **Validation Data**
- Passing `X_val` and `y_val` to the `train` method is optional. If not provided, the model will skip validation.

3. **Large Batch Sizes or Memory Constraints**
- Be mindful of batch sizes if using GPU-accelerated training with large datasets.

