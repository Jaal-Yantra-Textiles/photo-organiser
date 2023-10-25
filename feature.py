import tensorflow as tf
import numpy as np
import cv2
import tensorflow_hub as hub

from utility import logger

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch 

facenet = InceptionResnetV1(pretrained='vggface2').eval()

detector = MTCNN()


def extract_texture_features(feature_extractor, roi):
    """
    Extract texture features from the given Region Of Interest (ROI) using the provided feature_extractor.
    """
    # Convert the ROI to grayscale
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)  # Convert back to BGR format for consistency with the feature extractor input

    # Resize and preprocess the grayscale ROI
    roi_resized = tf.image.resize(roi_gray, [224, 224])
    roi_resized = tf.cast(roi_resized, dtype=tf.float32)  # Convert to float32
    roi_resized = tf.expand_dims(roi_resized, axis=0)

    # Use the feature extractor to get texture features
    texture_features = feature_extractor(roi_resized)
    
    return texture_features.numpy()

def extract_pattern_features(feature_extractor, roi):
    """
    Extract pattern features from the given Region Of Interest (ROI) using the provided feature_extractor.
    """
    # Apply edge detection (using Canny edge detector for demonstration)
    roi_edges = cv2.Canny(roi, 100, 200)
    roi_edges_colored = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)  # Convert to BGR format for consistency

    # Resize and preprocess the edge-detected ROI
    roi_resized = tf.image.resize(roi_edges_colored, [224, 224])
    roi_resized = tf.cast(roi_resized, dtype=tf.float32)  # Convert to float32
    roi_resized = tf.expand_dims(roi_resized, axis=0)

    # Use the feature extractor to get pattern features
    pattern_features = feature_extractor(roi_resized)
    
    return pattern_features.numpy()


def get_person_bboxes_with_high_confidence(detections):
    """
    Get all bounding boxes of the 'person' class with a confidence level above 0.99.
    """
    boxes = detections['detection_boxes'][0].numpy()
    scores = detections['detection_scores'][0].numpy()
    PERSON_CLASS_INDEX = 1
    class_labels = detections['detection_classes'][0].numpy()

    high_confidence_boxes = []

    for i, (box, score, label) in enumerate(zip(boxes, scores, class_labels)):
        if label == PERSON_CLASS_INDEX and score >= 0.90:
            high_confidence_boxes.append(box)
    
    return high_confidence_boxes


def get_face_from_roi(roi):
    """
    Extracts the face from the given ROI (Region of Interest).
    """
    bbox = get_face_bbox(roi)
    if bbox is not None:
        actual_bbox = bbox[0]
        logger.info(f"Actual bounding box: {actual_bbox}")
        face_roi = roi[int(actual_bbox[1]):int(actual_bbox[3]), int(actual_bbox[0]):int(actual_bbox[2])]
        logger.info(f"Shape of face_roi: {face_roi.shape}")  # Add this
        return face_roi
    return None

def get_face_bbox(img):
    faces = detector.detect(img)
    if faces is None or len(faces) > 0:
        bbox = faces[0]
        return bbox
    return None


def get_face_embedding(face_img):
    """
    Get the embedding of a face using FaceNet.
    """
    if face_img.size == 0:
        logger.warning("Detected face image has zero size. Skipping resizing and embedding.")
        return None
    if len(face_img.shape) == 3:
        face_img = tf.expand_dims(face_img, axis=0)
    # Resize the image to the size expected by FaceNet (160, 160)
    face_resized = tf.image.resize(face_img, [160, 160])
    face_resized = face_resized / 255.0  # Normalize to [0, 1]
    
    # Convert the TensorFlow tensor to a numpy array
    face_resized_np = face_resized.numpy()
    
    # Convert the numpy array to a PyTorch tensor
    input_tensor = torch.from_numpy(face_resized_np).permute(0, 3, 1, 2).float()
    
    # Get embedding
    with torch.no_grad():  # Ensure no gradients are computed
        embedding = facenet(input_tensor)
        
    return embedding.numpy()


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))


