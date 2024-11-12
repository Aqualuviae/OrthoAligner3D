# OrthoAligner3D

OrthoAligner3D is an automated 3D model design tool tailored for creating alignment splints in orthodontics. This project utilizes machine learning to streamline the design of 3D models based on STL files, facilitating the automated creation of alignment splints for orthodontic treatment.

## Project Structure

- **DataPreparation.py**: Manages initial data processing tasks. This script assumes STL files are available in the `ModelData` directory.
- **FeatureExtraction.py**: Extracts and organizes feature data from STL files, preparing it for model training.
- **TrainModel.py**: Defines and trains a neural network to predict 3D model structures, saving the trained model and plotting training loss.
- **DesignWithModel.py**: Uses the trained model to generate new 3D designs, producing STL files for alignment splints based on input data.
- **main.py**: The primary entry point, allowing the user to choose between training a model and generating designs with a pre-trained model.

## Features

1. **Data Preparation**:
   - Organizes STL files for model training, ensuring compatibility for downstream tasks.

2. **Feature Extraction**:
   - Extracts critical 3D structural features from `AntagonistScan.stl` and `PreparationScan.stl` files.
   - Structures the data to optimize training for predictive model generation.

3. **Model Training**:
   - Utilizes a neural network model to train on STL-based 3D data features, learning to predict accurate aligner splint models.
   - Saves the trained model and provides a training loss plot for performance assessment.

4. **Automated 3D Model Generation**:
   - Based on the trained model, generates STL files representing alignment splints for orthodontic use.
   - Processes new patient scans and generates corresponding splint designs, facilitating efficient orthodontic workflows.

## Installation

1. **Prerequisites**:
   - Python 3.8 or later
   - Required libraries: `tensorflow`, `numpy`, `open3d`, `matplotlib`

   Install the dependencies with:
   ```bash
   pip install tensorflow numpy open3d matplotlib
