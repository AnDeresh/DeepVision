from .random_forest_classifier import RandomForestMnistClassifier
from .ffnn_classifier import FfnnMnistClassifier
from .cnn_classifier import CnnMnistClassifier

# Initialize the MNIST classifier wrapper
class MnistClassifier:
    def __init__(self, algorithm, input_shape=None, num_classes=10, **kwargs):
        """
        Initialize the MNIST classifier wrapper.
        
        Parameters:
        - algorithm: A string specifying which model to use. Options are:
            'rf'  - Random Forest
            'nn'  - Feed-Forward Neural Network (FFNN)
            'cnn' - Convolutional Neural Network (CNN)
        - input_shape: Tuple representing the shape of the input data.
                       Required for 'nn' and 'cnn'. For MNIST, typically (28, 28) for FFNN and (28, 28, 1) for CNN.
        - num_classes: Integer representing the number of output classes (default is 10 for MNIST).
        - **kwargs: Additional keyword arguments that are passed to the model initializer.
        """
        if algorithm == 'rf':
            # Initialize the Random Forest classifier
            self.model = RandomForestMnistClassifier(**kwargs)
        elif algorithm == 'nn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for FFNN")
            # Initialize the Feed-Forward Neural Network classifier
            self.model = FfnnMnistClassifier(input_shape, num_classes)
        elif algorithm == 'cnn':
            if input_shape is None:
                raise ValueError("input_shape must be provided for CNN")
            # Initialize the Convolutional Neural Network classifier
            self.model = CnnMnistClassifier(input_shape, num_classes)
        else:
            raise ValueError("Unsupported algorithm. Choose from 'rf', 'nn', or 'cnn'.")

    def preprocess(self, X):
        """
        Preprocess the data using the underlying model's preprocess method.
        
        Parameters:
        - X: Input data (images).
        
        Returns:
        - Preprocessed data.
        """
        return self.model.preprocess(X)

    def train(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Train the underlying model.
        
        Parameters:
        - X_train: Training images.
        - y_train: Training labels.
        - X_val: (Optional) Validation images.
        - y_val: (Optional) Validation labels.
        - **kwargs: Additional keyword arguments (e.g., epochs, batch_size) for training.
        """
        self.model.train(X_train, y_train, X_val, y_val, **kwargs)

    def predict(self, X_test):
        """
        Predict labels for the provided test data using the underlying model.
        
        Parameters:
        - X_test: Test images.
        
        Returns:
        - Predicted labels.
        """
        return self.model.predict(X_test)

    def validate(self, X_val, y_val):
        """
        Validate the model on the validation data using the underlying model's validate method.
        
        Parameters:
        - X_val: Validation images.
        - y_val: Validation labels.
        
        Returns:
        - Accuracy score as a float.
        """
        return self.model.validate(X_val, y_val)