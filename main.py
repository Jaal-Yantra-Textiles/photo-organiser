import os
import concurrent.futures
import face_recognition
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.cluster import DBSCAN
import shutil
import signal

from PIL import Image

# Signal handling for graceful shutdown
shutdown = False

def signal_handler(signum, frame):
    global shutdown
    print("Signal received. Shutting down gracefully...")
    shutdown = True

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Load ResNet50 model for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False)

def extract_features(roi):
    # Convert the ROI array back to an image
    roi_image = Image.fromarray((roi * 255).astype(np.uint8))
    
    # Resize the image to the required dimensions
    roi_image = roi_image.resize((224, 224))
    
    x = image.img_to_array(roi_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = base_model.predict(x)
    return features.reshape((features.shape[1] * features.shape[2] * features.shape[3]))

def process_image(img_file):
    if shutdown:
        return None

    print(f"Processing {img_file}")
    image_np = face_recognition.load_image_file(img_file)
    face_locations = face_recognition.face_locations(image_np, model="cnn")
    
    if len(face_locations) == 1:
        top, right, bottom, left = face_locations[0]
        # Extract region below the face for clothing
        roi = image_np[bottom:bottom + (bottom - top), left:right]
        features = extract_features(roi)
        return ('single', img_file, features)
    elif len(face_locations) > 1:
        return ('multiple', img_file, None)
    else:
        return ('none', img_file, None)

image_folder = 'photos'
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.JPG')]

single_face_images = []
multiple_face_images = []
clothing_features = []

print("Processing images for face detection and clothing feature extraction...")
results = []
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_image, image_files))

for img_type, img_file, features in results:
    if img_type == 'single' and features is not None:
        single_face_images.append(img_file)
        clothing_features.append(features)
    elif img_type == 'multiple':
        multiple_face_images.append(img_file)

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Organize images with a single face based on clothing similarity
if len(clothing_features) > 0:
    print("Clustering images based on clothing similarity...")
    clothing_clusters = DBSCAN(eps=0.5, min_samples=5).fit(clothing_features)
    for idx, label in enumerate(clothing_clusters.labels_):
        cluster_dir = os.path.join(image_folder, f"clothing_cluster_{label}")
        create_directory(cluster_dir)
        shutil.copy(single_face_images[idx], cluster_dir)

# Organize images with multiple faces
multiple_faces_dir = os.path.join(image_folder, "multiple_faces")
create_directory(multiple_faces_dir)
for img_file in multiple_face_images:
    shutil.copy(img_file, multiple_faces_dir)

print("Organizing complete!")
