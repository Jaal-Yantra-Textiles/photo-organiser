import tensorflow as tf

from object_detection.utils import config_util , label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

import logging
import numpy as np

logger = logging.getLogger()


class BodyDetector:
    def __init__(self, model_checkpoint_path, pipeline_config_path, confidence_threshold=0.8, label_map_path=None):
       # Load the pipeline config and build the detection model
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        model_config = configs['model']
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(model_checkpoint_path).expect_partial()
        
        self.confidence_threshold = confidence_threshold
        
        # Load the label map if provided
        if label_map_path:
            self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)
        else:
            self.category_index = None
        
        logger.info(f"Model loaded: {self.detection_model}")
    
    

    def detect_body(self, img):
        try:
        # Log the shape of the input image
            logger.info(f"Shape of the image being processed: {img.shape}")

            # Convert the image to tensor
            input_tensor = tf.convert_to_tensor([img], dtype=tf.float32)

            # Directly get detections
            detections = self.detection_model(input_tensor)
            

            # Extract necessary information
            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            rawScores = detections['raw_detection_scores'][0].numpy()
            logger.info(f"Total Scores: {scores}, Raw Scores{rawScores}")
            classes = detections['detection_classes'][0].numpy()
            logger.info(f"Total Classes: {classes}")

            # Log details
            logger.info(f"Total detections: {len(scores)}")

            detected_objects = []
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    detected_object = {
                        "class": int(classes[i]),
                        "score": scores[i],
                        "box": boxes[i]
                    }
                    detected_objects.append(detected_object)
                    logger.info(f"Detection {i}: Class {classes[i]}, Score {scores[i]}, Box {boxes[i]}")

            if not detected_objects:
                logger.warning(f"No objects detected in the image above the confidence threshold.")
                return None

            return detected_objects

        except Exception as e:
            logger.error(f"Error during detection: {e}")
            return None
