from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from .mnist_classifier_interface import MnistClassifierInterface

# Implementing a class for Random Forest
class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self, **kwargs):
        # Initialize the Random Forest model with any given parameters
        self.model = RandomForestClassifier(**kwargs)
    
    def preprocess(self, X):
        """
        Preprocess the input data by normalizing pixel values and flattening the images.
        Assumes that X has shape (n_samples, 28, 28).
        """
        # Convert pixel values to float32 and normalize to the range [0, 1]
        X = X.astype('float32') / 255.0
        # Get the number of samples
        n_samples = X.shape[0]
        # Flatten each 28x28 image into a 784-length vector
        return X.reshape(n_samples, -1)
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model using training data.
        
        Parameters:
        - X_train: Training images.
        - y_train: Training labels.
        - X_val: (Optional) Validation images.
        - y_val: (Optional) Validation labels.
        """
        # Preprocess the training data
        X_train_processed = self.preprocess(X_train)
        # Train the model using the processed training data
        self.model.fit(X_train_processed, y_train)
        
        # If validation data is provided, evaluate the model's accuracy
        if X_val is not None and y_val is not None:
            acc = self.validate(X_val, y_val)
            print(f"Validation Accuracy: {acc:.4f}")
    
    def predict(self, X_test):
        """
        Predict labels for the provided test data.
        
        Parameters:
        - X_test: Test images.
        
        Returns:
        - Predicted labels for the test data.
        """
        # Preprocess the test data
        X_test_processed = self.preprocess(X_test)
        # Return the predictions from the model
        return self.model.predict(X_test_processed)
    
    def validate(self, X_val, y_val):
        """
        Evaluate the model on the validation data and return the accuracy.
        
        Parameters:
        - X_val: Validation images.
        - y_val: Validation labels.
        
        Returns:
        - Accuracy score as a float.
        """
        # Preprocess the validation data
        X_val_processed = self.preprocess(X_val)
        # Generate predictions for the validation data
        predictions = self.model.predict(X_val_processed)
        # Return the computed accuracy score
        return accuracy_score(y_val, predictions)