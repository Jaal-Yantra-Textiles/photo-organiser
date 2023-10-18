import os
from facenet_pytorch import MTCNN, InceptionResnetV1
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
import shutil
from feature import extract_features
import sqlite3
import logging
from skimage.color import rgb2hsv
from skimage.io import imread
import cv2

logging.basicConfig(filename='image_processing.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Connect to SQLite database (will create a new file if it doesn't exist)
conn = sqlite3.connect('image_features.db')
cursor = conn.cursor()


# Load model for dress feature extraction
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()



def store_features_in_db(img_path, face_features, dress_features):
    with sqlite3.connect('image_features.db') as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO image_features (filename, face_features, dress_features) VALUES (?, ?, ?)", 
                       (img_path, face_features.tobytes(), dress_features.tobytes()))
        conn.commit()

def process_single_face_image(img_file):
    try:
        logger.info(f"Processing single face image: {img_file}")
        img = cv2.cvtColor(cv2.imread("ivan.jpg"), cv2.COLOR_BGR2RGB)
        face_detections, _ = mtcnn.detect(img)

        if face_detections is None or len(face_detections) == 0:
            logger.warning(f"No face detected in {img_file}.")
            return None
        if len(face_detections) == 1:
            bounding_box = face_detections[0]
            left, top, right, bottom = bounding_box

            # Extract face embeddings using InceptionResnetV1
            face_crop = mtcnn(img, save_path=None)  # This crops the detected face
            face_encodings = resnet(face_crop.unsqueeze(0)).detach().numpy()

            # Use a body detection model to detect the body region
            body_detections = hypothetical_body_detector(img)  # This is a placeholder for actual body detection

            if body_detections:
                dress_left, dress_top, dress_right, dress_bottom = body_detections[0]
            else:
                # If body is not detected, use the previous method to estimate dress/body bounding box
                face_height = bottom - top
                dress_top = bottom
                dress_bottom = bottom + 6 * face_height  # Assuming body is roughly 6 times the face height
                dress_left = left
                dress_right = right

            # Ensure the bounding box coordinates are within image boundaries
            dress_top = max(0, dress_top)
            dress_bottom = min(img.shape[0], dress_bottom)  # Use img.shape instead of img.size for numpy arrays
            dress_left = max(0, dress_left)
            dress_right = min(img.shape[1], dress_right)

            # Extract dress/body region
            dress_roi = np.array(img)[int(dress_top):int(dress_bottom), int(dress_left):int(dress_right)]
            dress_features = extract_features(dress_roi)

            store_features_in_db(img_file, face_encodings, dress_features)
            return face_encodings, dress_features
        else:
            logger.warning(f"Multiple faces detected in {img_file}. Skipping.")
            return None
    except Exception as e:
        logger.error(f"Error processing {img_file}. Error: {e}")
        return None
    
def extract_color_histogram(image, bins=(8, 8, 8)):
    """Extract a 3D color histogram from the RGB image."""
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def cluster_and_organize_images():
    with sqlite3.connect('image_features.db') as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT filename, face_features, dress_features FROM image_features")
        data = cursor.fetchall()

    single_face_image_files = [row[0] for row in data]
    face_features_list = [np.frombuffer(row[1], dtype=np.float32) for row in data]

    # Extract color histograms for dress regions
    dress_color_features_list = [extract_color_histogram(imread(filename)) for filename in single_face_image_files]

    # Cluster based on face features
    face_clusters = DBSCAN(eps=1, min_samples=10).fit(face_features_list)
    for face_label in set(face_clusters.labels_):
        face_cluster_indices = [idx for idx, label in enumerate(face_clusters.labels_) if label == face_label]
        face_cluster_files = [single_face_image_files[idx] for idx in face_cluster_indices]
        face_cluster_colors = [dress_color_features_list[idx] for idx in face_cluster_indices]

        # Cluster this face cluster based on dress color features using KMeans
        kmeans = KMeans(n_clusters=10)  # Adjust the number of clusters if needed
        dress_clusters = kmeans.fit_predict(face_cluster_colors)
        for dress_label in set(dress_clusters):
            dress_cluster_indices = [idx for idx, label in enumerate(dress_clusters) if label == dress_label]
            for idx in dress_cluster_indices:
                cluster_dir = os.path.join('photos', f"face_{face_label}_dress_{dress_label}")
                if not os.path.exists(cluster_dir):
                    os.makedirs(cluster_dir)
                shutil.copy(face_cluster_files[idx], cluster_dir)

    logger.info("Organizing single face images complete!")