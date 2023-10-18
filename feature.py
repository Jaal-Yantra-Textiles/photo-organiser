from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from skimage.feature import local_binary_pattern, gabor
from PIL import Image
import numpy as np
import cv2

# Load ResNet50 model for pattern feature extraction
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
pattern_extraction_model = Model(inputs=base_model.input, outputs=x)

def extract_color_histogram(roi, bins=(8, 8, 8)):
    """Extract a 3D color histogram from the RGB image."""
    hist = cv2.calcHist([roi], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_pattern_features(roi):
    # Convert the ROI array back to an image
    roi_image = Image.fromarray((roi * 255).astype(np.uint8))
    
    # Resize the image to the required dimensions
    roi_image = roi_image.resize((224, 224))
    
    x = image.img_to_array(roi_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    features = pattern_extraction_model.predict(x)
    
    # Normalize the feature vector to have a unit length
    features = features / np.linalg.norm(features)
    
    return features[0]

def extract_texture_features(roi):
    # Convert to grayscale as LBP and Gabor work on single channel
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # LBP features
    lbp = local_binary_pattern(gray, P=24, R=8, method="uniform")
    (lbp_hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    lbp_hist = lbp_hist.astype("float")
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize
    
    # Gabor features
    gabor_features = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 4): # 4 orientations
        for sigma in [1, 3]: # 2 scales
            kern = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
            temp = cv2.filter2D(gray, cv2.CV_8UC3, kern)
            gabor_features.append(np.mean(temp))
            gabor_features.append(np.std(temp))
    
    return np.concatenate([lbp_hist, gabor_features])

def extract_features(roi):
    # Extract color features
    color_features = extract_color_histogram(roi)
    
    # Extract pattern features
    pattern_features = extract_pattern_features(roi)
    
    # Extract texture features
    texture_features = extract_texture_features(roi)
    
    # Concatenate and return the combined feature vector
    combined_features = np.concatenate([color_features, pattern_features, texture_features])
    
    return combined_features
