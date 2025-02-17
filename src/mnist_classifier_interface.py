from abc import ABC, abstractmethod

# Creating a basic interface
class MnistClassifierInterface(ABC):
    @abstractmethod
    def preprocess(self, X):
        """
        Preprocess the input data.
        This could include normalization, reshaping, or any custom transformation.
        """
        pass

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on the training data.
        
        Parameters:
        - X_train, y_train: training data and labels.
        - X_val, y_val: optional validation data and labels.
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Predict the labels for the test data.
        """
        pass

    @abstractmethod
    def validate(self, X_val, y_val):
        """
        Evaluate the model on the validation data and return evaluation metrics.
        """
        pass