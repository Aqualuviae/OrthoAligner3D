import os
import numpy as np
import open3d as o3d
from FeatureExtraction import load_stl_file


def generate_models (design_data_dir, model):
    """ Use the pre-trained model to generate designs based on new STL files """
    for case in os.listdir(design_data_dir):
        case_dir = os.path.join(design_data_dir, case)
        if os.path.isdir(case_dir):
            print(f"Generating models for case {case}")

            # Load the scans
            scans = ['AntagonistScan.stl', 'PreparationScan.stl']
            scan_features = np.hstack([load_stl_file(os.path.join(case_dir, f)) for f in scans]).reshape(1, -1)

            # Predict the output using the pre-trained model
            output = model.predict(scan_features)

            for i in range(2):  # We assume there are 2 models to generate
                print(f"Processing model {i + 1} for case {case}")

                try:
                    vertices = output[0, i * 2499:(i + 1) * 2499].reshape(-1, 3)

                    # Create dummy triangles for demonstration purposes
                    triangles = np.array([[j, j + 1, j + 2] for j in range(0, len(vertices) - 2, 3)])

                    mesh = o3d.geometry.TriangleMesh(
                        vertices=o3d.utility.Vector3dVector(vertices),
                        triangles=o3d.utility.Vector3iVector(triangles)
                    )

                    # Save the output meshes
                    output_path_1 = os.path.abspath(os.path.join(case_dir, f"{i + 1}.stl"))
                    o3d.io.write_triangle_mesh(output_path_1, mesh)
                    print(f"Generated model {i + 1} for case {case}")

                except ValueError as e:
                    print(f"Error processing model {i + 1} for case {case}: {e}")
