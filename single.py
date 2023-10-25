import os
from body import BodyDetector
from facenet_pytorch import MTCNN, InceptionResnetV1
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
from feature import extract_features
import sqlite3
import logging
from skimage.color import rgb2hsv
from skimage.io import imread
import cv2
import tensorflow as tf
from PIL import Image

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

tf.debugging.experimental.enable_dump_debug_info(
    "log_dir",
    tensor_debug_mode="FULL_HEALTH",
    circular_buffer_size=-1
)


logger = logging.getLogger()

# Connect to SQLite database (will create a new file if it doesn't exist)
conn = sqlite3.connect('image_features.db')
cursor = conn.cursor()


# Load model for dress feature extraction
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()

body_detector = BodyDetector(
    model_checkpoint_path='faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8/checkpoint/ckpt-0',
    pipeline_config_path='faster_rcnn_resnet50_v1_1024x1024_coco17_tpu-8/pipeline.config',
    confidence_threshold=0.85
)

def store_features_in_db(img_path, face_features, dress_features):
    try:
        with sqlite3.connect('image_features.db') as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO image_features (filename, face_features, dress_features) VALUES (?, ?, ?)", 
                           (img_path, face_features.tobytes(), dress_features.tobytes()))
            conn.commit()
        logger.info(f"Stored features for {img_path} in database.")
    except Exception as e:
        logger.error(f"Error storing features for {img_path}. Error: {e}")


def process_single_face_image(img_file):
    try:
        logger.info(f"Loading image: {img_file}")
        # Use PIL to open the image
        img_pil = np.array(Image.open(img_file))
    
        
        logger.info(f"Detecting faces in image: {img_file}")
        face_detections, _ = mtcnn.detect(img_pil)

        if face_detections is None or len(face_detections) == 0:
            logger.warning(f"No face detected in {img_file}.")
            return None

        # List to hold features of all detected faces and dresses in the image
        all_features = []

        for face_box in face_detections:
            features = process_face_and_dress(img_pil, face_box, img_file)
            all_features.append(features)

        return all_features

    except Exception as e:
        logger.error(f"Error processing {img_file}. Error: {e}")
        return None

def process_face_and_dress(img, bounding_box, img_file):
    try:
        left, top, right, bottom = bounding_box

        # Extract face embeddings using InceptionResnetV1
        logger.info(f"Extracting face embeddings for {img_file}")
        face_crop = mtcnn(img, save_path=None)
        # Check if face_crop is None
        if face_crop is None:
            logger.warning(f"No face cropped in {img_file}.")
            return None
        face_encodings = resnet(face_crop.unsqueeze(0)).detach().numpy()

        # Use Faster R-CNN for body detection
        logger.info(f"Detecting body using Faster R-CNN for {img_file}")
        dress_box = body_detector.detect_body(img)
        if not dress_box:
            logger.warning(f"No body detected for face in {img_file}. Using face-based estimation.")
            face_height = bottom - top
            dress_top = bottom
            dress_bottom = bottom + 6 * face_height
        else:
            dress_left, dress_top, dress_right, dress_bottom = dress_box

        # Extract dress/body region
        logger.info(f"Extracting dress/body features for {img_file}")
        dress_roi = np.array(img)[int(dress_top):int(dress_bottom), int(dress_left):int(dress_right)]
        dress_features = extract_features(dress_roi)

        # Store face and dress features in the database
        store_features_in_db(img_file, face_encodings, dress_features)

        return face_encodings, dress_features

    except Exception as e:
        logger.error(f"Error processing face and dress for {img_file}. Error: {e}")
        return None
