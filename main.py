import os
import TrainModel
import DesignWithModel
import FeatureExtraction
from tensorflow.keras.models import load_model

def main():
    design_data_dir = "D:/Software/Tool/Programming/Python/Python 3.11/pythonproject/3ShapeModel/DesignData"
    model_data_dir = "D:/Software/Tool/Programming/Python/Python 3.11/pythonproject/3ShapeModel/ModelData"
    model_path = "D:/Software/Tool/Programming/Python/Python 3.11/pythonproject/3ShapeModel/trained_model.h5"

    mode = input("Enter 'train' to train model, 'design' to use existing model: ")

    if mode == 'train':
        # Extract features and labels for training
        features, labels = FeatureExtraction.extract_features(model_data_dir)
        print(f"Feature dimensions: {features.shape}, Label dimensions: {labels.shape}")
        # Train the model
        TrainModel.train_model(features, labels)
    elif mode == 'design':
        # Load pre-trained model
        print("Loaded pre-trained model.")
        model = load_model(model_path)
        # Use the model to generate designs
        DesignWithModel.generate_models(design_data_dir, model)
    else:
        print("Invalid option. Please enter 'train' or 'design'.")

if __name__ == '__main__':
    main()
