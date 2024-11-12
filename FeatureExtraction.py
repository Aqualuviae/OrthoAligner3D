import os
import numpy as np
import open3d as o3d


def load_stl_file (file_path):
    """ Load the vertices from an STL file """
    mesh = o3d.io.read_triangle_mesh(file_path)
    return np.asarray(mesh.vertices).flatten()


def extract_features (model_data_dir):
    """ Extract features and labels from STL files in ModelData """
    features = []
    labels = []

    for case in os.listdir(model_data_dir):
        case_dir = os.path.join(model_data_dir, case)
        if os.path.isdir(case_dir):
            # Assuming AntagonistScan.stl is the feature and PreparationScan.stl is the label
            antagonist_scan = load_stl_file(os.path.join(case_dir, 'AntagonistScan.stl'))
            preparation_scan = load_stl_file(os.path.join(case_dir, 'PreparationScan.stl'))

            features.append(antagonist_scan)
            labels.append(preparation_scan)

    # Convert lists to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    return features, labels
