# Jaal Yantra: Automating Fashion with Speed

Just run `python enq.py process photos` 


## Introduction
At **Jaal Yantra**, we are not just another textile company. We are an embodiment of passion and dedication, focused on revolutionizing the fashion industry by merging traditional designs with futuristic technology. Our vision is to bring the world fashion with unparalleled speed and precision. Like any major textile company, we are inundated with an array of vibrant and intricate photos that capture the essence of our brand. But in this vast sea of images, we saw an opportunity: an opportunity to innovate, automate, and accelerate our processes.

## Features
1. Multithreaded Processing:

The tool uses Python's threading and queue modules to create a pool of worker threads. This allows it to process multiple images concurrently, significantly speeding up batch operations.

2. Image Processing:

Process images to detect faces and extract features. Specifically designed for images with a single face.

3. Clustering and Organizing:

Cluster processed images based on extracted features and organize them accordingly. This aids in categorization and tagging.

4. Image Conversion:

Convert images from .JPG format to .JPEG format, aiding in standardization.

5. Sample Model Execution:

Test your system's GPU capability by running a sample deep learning model.

## The Challenge
The digital age brings with it a deluge of images. For a fashion-forward company like ours, these aren't just images; they are a reflection of our brand, our designs, and our story. But with thousands of images, how do we efficiently categorize, tag, and upload them to our e-commerce platform? Manual categorization is not only time-consuming but also prone to errors.

## The Solution: Jaal Yantra's Image Processing Tool
Enter our state-of-the-art image processing tool. This tool is not just about identifying images; it's about understanding them. It can:
- **Identify Faces**: Recognize and differentiate between single and multiple faces in images.
- **Feature Extraction**: Delve deeper into the images to extract unique features from the dresses.
- **Classification**: Classify images based on the extracted features, ensuring a streamlined categorization process.
- **Automation for E-commerce**: The final step is to seamlessly upload these categorized and tagged images to our e-commerce platform, ready for our customers to experience.

## Tech Stack

### Core Technologies:
- **Python**: The backbone of our tool, providing flexibility and robustness.
- **TensorFlow**: Leveraged for deep learning, enabling accurate image and face recognition.
- **TensorFlow Object Detection API**: Facilitates the use of pre-trained models like Faster R-CNN for body detection.
- **OpenCV**: For image processing tasks, such as color conversion and histogram extraction.
- **SQLite**: Lightweight database to store and manage image features.

### Deep Learning Models:
- **MTCNN**: For accurate face detection.
- **InceptionResnetV1**: Used for face embeddings.
- **ResNet50**: Assists in extracting deep features from dresses.
- **Faster R-CNN**: Enables precise body detection, ensuring we focus on the right parts of the image.

### Additional Libraries:
- **PIL (Python Imaging Library)**: Aids in image manipulation tasks.
- **Scikit-learn**: Used for clustering the images based on their features.
- **Scikit-image**: Helps in converting images and extracting texture features.

## Conclusion
Jaal Yantra is more than just a name; it's a symbol of innovation in the fashion industry. With our cutting-edge image processing tool, we are set to redefine the way textile companies view and manage their images. Join us on this exciting journey as we weave technology into the fabric of fashion, one image at a time.