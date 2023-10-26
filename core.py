import os
from utility import serialize_array
import numpy as np
from sqlite_utility import SQLiteDB

import PIL as PIL
from PIL import  Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np


import cv2

from utility import deserialize_array

import tensorflow as tf

import tensorflow_hub as hub


from feature import extract_pattern_features, extract_texture_features , get_person_bboxes_with_high_confidence,get_face_from_roi, get_face_embedding, euclidean_distance
from logger_util import logger

# Model related path
LOCAL_MODEL_PATH = 'model-using'
LOCAL_MODEL_PATH2 = '5'
model = tf.saved_model.load(LOCAL_MODEL_PATH)
feature_extractor = hub.KerasLayer(LOCAL_MODEL_PATH2, trainable=False)

db_path = "images_comparison.db"

db_util = SQLiteDB(db_path)

def extract_and_save_features(directory_path):
    """
    Extract features from all images in the given directory and save them in the SQLite database.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"'{directory_path}' is not a directory.")
        raise ValueError(f"'{directory_path}' is not a directory.")
    
    logger.info(f"Starting feature extraction for images in directory: {directory_path}")
    
    # Loop through all images in the directory
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        logger.info(f"Starting feature extraction for  {image_path}")
        # Load and preprocess the image
        img_np = load_image_into_numpy_array(image_path)
        img_np_processed = preprocess_image(img_np.copy())

        # Detect person and extract ROI
        input_tensor = tf.convert_to_tensor([img_np_processed], dtype=tf.uint8)
        detections = model.signatures['serving_default'](input_tensor)
        box = get_person_bboxes_with_high_confidence(detections)

        if box is not None:
            roi = img_np[int(box[0]*img_np.shape[0]):int(box[2]*img_np.shape[0]), int(box[1]*img_np.shape[1]):int(box[3]*img_np.shape[1])]

            # Extract face embedding
            face_features = get_face_from_roi(roi)
            face_embedding = get_face_embedding(face_features) if face_features is not None else None

            # Extract other features (using the same function for now as placeholders)
            roi_resized = tf.image.resize(roi, [224, 224])
            roi_resized = tf.cast(roi_resized, dtype=tf.float32)  # Convert to float32
            roi_resized = tf.expand_dims(roi_resized, axis=0)
            features = feature_extractor(roi_resized)
            textures = extract_texture_features(feature_extractor, roi)
            patterns = extract_pattern_features(feature_extractor, roi)

            # Save the features in the SQLite database
            db_util.insert_image_features(image_path, face_embedding, features, textures, patterns)
            logger.info(f"Features extracted and saved for image: {image_path}")
        else:
            logger.warning(f"No bounding box detected for image: {image_path}")
            
        db_util.close()
    
    logger.info(f"Feature extraction completed for directory: {directory_path}")

def compare_all_images():
    """
    Compare features of all images with each other and save the comparison scores in the SQLite database.
    """
    logger.info("Starting image comparison process")
    
    all_features = db_util.fetch_all_image_features()


    # Compare each image with every other image
    for i, (id1, _, face_embedding1, color_feature1, texture_feature1, pattern_feature1, _) in enumerate(all_features):
        for j, (id2, _, face_embedding2, color_feature2, texture_feature2, pattern_feature2, _) in enumerate(all_features):
            if i < j:  # Ensure we don't compare an image with itself and avoid redundant comparisons
                # Compute face distance
                if face_embedding1 is not None and face_embedding2 is not None:
                    face_distance = np.linalg.norm(deserialize_array(face_embedding1) - deserialize_array(face_embedding2))
                else:
                    face_distance = None  # Set to None if face embedding is missing for any image

                # Compute other feature distances (using Euclidean distance for demonstration)
                color_distance = euclidean_distance(deserialize_array(color_feature1), deserialize_array(color_feature2))
                texture_distance = euclidean_distance(deserialize_array(texture_feature1), deserialize_array(texture_feature2))
                pattern_distance = euclidean_distance(deserialize_array(pattern_feature1), deserialize_array(pattern_feature2))

                db_util.insert_comparison_scores(id1, id2, face_distance, color_distance, texture_distance, pattern_distance)
                logger.info(f"Comparison scores saved for image IDs: {id1} and {id2}")
                
        db_util.close()
    
    logger.info("Image comparison process completed")




def detect_and_store_face(directory_path):
    """
    Detect faces in all images in the given directory and save the ROI in the SQLite database.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"'{directory_path}' is not a directory.")
        raise ValueError(f"'{directory_path}' is not a directory.")
    
    logger.info(f"Starting face detection for images in directory: {directory_path}")
    
    # Loop through all images in the directory
    for filename in os.listdir(directory_path):
        image_path = os.path.join(directory_path, filename)
        logger.info(f"Detecting face in {image_path}")
        try:

        # Load and preprocess the image
            img_np = load_image_into_numpy_array(image_path)
            img_np_processed = preprocess_image(img_np.copy())

            # Detect person and extract ROI
            input_tensor = tf.convert_to_tensor([img_np_processed], dtype=tf.uint8)
            detections = model.signatures['serving_default'](input_tensor)
            boxes = get_person_bboxes_with_high_confidence(detections)

            if boxes:
                for box in boxes:
                    roi = img_np[int(box[0]*img_np.shape[0]):int(box[2]*img_np.shape[0]), int(box[1]*img_np.shape[1]):int(box[3]*img_np.shape[1])]
                    db_util.insert_image_roi(image_path, serialize_array(roi))
                logger.info(f"{len(boxes)} faces detected and ROIs saved for image: {image_path}")
            else:
                logger.warning(f"No bounding box detected for image: {image_path}")
                
        except PIL.UnidentifiedImageError:
            logger.warning(f"Unable to process {image_path}. Not a recognized image format.")
            continue

        db_util.close()

    logger.info(f"Face detection completed for directory: {directory_path}")


def extract_features_from_stored_roi():
    """
    Extract features from the stored ROI in the SQLite database.
    """
    
    all_images = db_util.fetch_all_image_rois()
    logger.info(f"Stored ROI Extraction for the images size {len(all_images)}")

    for image_path, serialized_roi in all_images:
        roi = deserialize_array(serialized_roi)
        

        # Extract face embedding
        face_features = get_face_from_roi(roi)
        face_embedding = get_face_embedding(face_features) if face_features is not None else None

        # Extract other features
        roi_resized = tf.image.resize(roi, [224, 224])
        roi_resized = tf.cast(roi_resized, dtype=tf.float32)  # Convert to float32
        roi_resized = tf.expand_dims(roi_resized, axis=0)
        features = feature_extractor(roi_resized)
        textures = extract_texture_features(feature_extractor, roi)
        patterns = extract_pattern_features(feature_extractor, roi)

        # Update the features in the SQLite database
        db_util.update_image_features(image_path, face_embedding, features, textures, patterns)
    
    logger.info(f"Stored ROI Extraction for the images finshed")

def load_image_into_numpy_array(path):
    """Loads an image from file into a numpy array."""
    image = Image.open(path)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def preprocess_image(np_image):
    """Preprocesses the image for detection."""
    # Resize the image such that the shortest side is 600px
    h, w, _ = np_image.shape
    scale = 600 / min(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)
    resized_image = cv2.resize(np_image, (new_w, new_h))
    
    return resized_image





def compare_all_images_stats(num_clusters=10):
    """
    Cluster images based on their features and store the cluster labels in the SQLite database.
    """
    logger.info("Starting image comparison process")

    all_features = db_util.fetch_all_image_features()

    # Extract features 
    color_features = [deserialize_array(f[3]).reshape(-1) for f in all_features]
    texture_features = [deserialize_array(f[4]).reshape(-1) for f in all_features]
    pattern_features = [deserialize_array(f[5]).reshape(-1) for f in all_features]

    # Assuming color_features is the longest list (or use any of the other lists)
    num_images = len(color_features)

    embeddings = []
    for f in all_features:
        if f[2] is not None:
            embeddings.append(deserialize_array(f[2]).reshape(-1))
        else:
            # Add a placeholder value. Here, I'm using zeros, but you can choose any other value.
            embeddings.append(np.zeros(512))

    # Convert lists to 2D numpy arrays
    embeddings = np.array(embeddings)
    color_features = np.array(color_features)
    texture_features = np.array(texture_features)
    pattern_features = np.array(pattern_features)

    # Combine all features into a single feature vector (this can be adjusted based on importance of features)
    combined_features = np.hstack((embeddings, color_features, texture_features, pattern_features))

    # Reduce dimensionality
    pca = PCA(n_components=50)  # Adjust based on desired number of components
    reduced_features = pca.fit_transform(combined_features)

    # Cluster images based on reduced features
    kmeans = KMeans(n_clusters=num_clusters)
    cluster_labels = kmeans.fit_predict(reduced_features)

    # Store cluster labels in the database for each image
    for i, (image_id, _, _, _, _, _, _) in enumerate(all_features):
        db_util.insert_cluster_label(image_id, int(cluster_labels[i]), reduced_features[i][0], reduced_features[i][1])
        logger.info(f"Cluster label and PCA features saved for image ID: {image_id, cluster_labels[i]}")


    db_util.close()
    logger.info("Image comparison process completed")




def generate_scatter_plot_image():
    image_path="scatter_plot.png"
    
    data = db_util.fetch_clustered_data_with_features()
    reduced_features = np.array([[row[2], row[3]] for row in data])
    cluster_labels = [row[1] for row in data]
    
    cluster_labels = np.array(cluster_labels, dtype=int)
    norm_cluster_labels = (cluster_labels - cluster_labels.min()) / (cluster_labels.max() - cluster_labels.min())
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=norm_cluster_labels, cmap='rainbow', s=50)
    plt.colorbar()
    plt.title('2D PCA of Image Features with Cluster Labels')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

    return image_path


