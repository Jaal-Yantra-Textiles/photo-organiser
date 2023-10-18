import os
import face_recognition
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
from main import extract_features
import concurrent.futures

def process_multiple_face_image(img_file):
    face_encodings = face_recognition.face_encodings(face_recognition.load_image_file(img_file))
    if len(face_encodings) > 1:
        # Extract dress features
        dress_features = extract_features(img_file)
        # Combine face and dress features for each face
        combined_features_list = [np.concatenate((face_encoding, dress_features)) for face_encoding in face_encodings]
        return combined_features_list
    else:
        return []

multiple_face_folder = 'multiple_faces'
multiple_face_image_files = [os.path.join(multiple_face_folder, f) for f in os.listdir(multiple_face_folder) if f.endswith('.JPG')]

print(f"Total multiple face images to process: {len(multiple_face_image_files)}")

with concurrent.futures.ThreadPoolExecutor() as executor:
    multiple_face_combined_features_results = list(executor.map(process_multiple_face_image, multiple_face_image_files))

multiple_face_combined_features_list = [features for sublist in multiple_face_combined_features_results for features in sublist]

print(f"Processed images with {len(multiple_face_combined_features_list)} multiple faces.")

# Clustering and organizing multiple face images
if len(multiple_face_combined_features_list) > 0:
    print("Clustering multiple face images based on face and dress similarity...")
    multiple_face_clusters = DBSCAN(eps=0.5, min_samples=5).fit(multiple_face_combined_features_list)
    for idx, label in enumerate(multiple_face_clusters.labels_):
        cluster_dir = os.path.join(multiple_face_folder, f"combined_cluster_{label}")
        if not os.path.exists(cluster_dir):
            os.makedirs(cluster_dir)
        shutil.copy(multiple_face_image_files[idx], cluster_dir)

print("Organizing multiple face images complete!")
