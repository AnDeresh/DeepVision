import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

from .mnist_classifier_interface import MnistClassifierInterface

# Implementation of a class for Convolutional Neural Network (CNN)
class CnnMnistClassifier(MnistClassifierInterface):
    def __init__(self, input_shape, num_classes):
        """
        Initialize the Convolutional Neural Network (CNN) model.
        
        Parameters:
        - input_shape: Tuple representing the shape of the input (e.g., (28, 28, 1)).
        - num_classes: Integer representing the number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        # Define the CNN architecture
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    
    def preprocess(self, X):
        """
        Preprocess the input data by normalizing pixel values and reshaping if necessary.
        Assumes that X has shape (n_samples, height, width) and converts it to 
        (n_samples, height, width, 1) for grayscale images.
        """
        # Convert to float32 and normalize
        X = X.astype('float32') / 255.0
        # Check if X has 3 dimensions; if so, expand to add channel dimension
        if len(X.shape) == 3:
            X = np.expand_dims(X, axis=-1)
        return X
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=10, batch_size=32):
        """
        Train the CNN model using training data.
        
        Parameters:
        - X_train: Training images.
        - y_train: Training labels.
        - X_val: (Optional) Validation images.
        - y_val: (Optional) Validation labels.
        - epochs: Number of training epochs.
        - batch_size: Batch size for training.
        """
        # Preprocess training data
        X_train_processed = self.preprocess(X_train)
        # Convert labels to one-hot encoding
        y_train_cat = to_categorical(y_train, self.num_classes)
        
        # Train the model
        self.model.fit(X_train_processed, y_train_cat, epochs=epochs, batch_size=batch_size, verbose=1)
        
        # If validation data is provided, evaluate the model
        if X_val is not None and y_val is not None:
            acc = self.validate(X_val, y_val)
            print(f"Validation Accuracy: {acc:.4f}")
    
    def predict(self, X_test):
        """
        Predict labels for the provided test data.
        
        Parameters:
        - X_test: Test images.
        
        Returns:
        - Predicted class labels for the test data.
        """
        X_test_processed = self.preprocess(X_test)
        # Get prediction probabilities and return the class with the highest probability
        predictions = self.model.predict(X_test_processed)
        return predictions.argmax(axis=1)
    
    def validate(self, X_val, y_val):
        """
        Evaluate the model on the validation data and return the accuracy.
        
        Parameters:
        - X_val: Validation images.
        - y_val: Validation labels.
        
        Returns:
        - Accuracy score as a float.
        """
        X_val_processed = self.preprocess(X_val)
        predictions = self.model.predict(X_val_processed)
        predicted_labels = predictions.argmax(axis=1)
        return accuracy_score(y_val, predicted_labels)