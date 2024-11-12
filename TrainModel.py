import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def train_model(features, labels):
    """ Train a neural network model with the given features and labels """
    # Define the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(features.shape[1],)),
        Dense(64, activation='relu'),
        Dense(labels.shape[1])
    ])

    model.compile(optimizer='adam', loss='mse')

    # Train the model
    history = model.fit(features, labels, validation_split=0.2, epochs=50)

    # Save the trained model
    model.save("path/to/saved_model.h5")

    # Plot and save the loss curve
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('training_loss.png')
    plt.show()

    print("Training completed. Loss curve saved as 'training_loss.png'.")
