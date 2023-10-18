import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

class BodyDetector:
    def __init__(self, model_checkpoint_path, pipeline_config_path):
        """
        Initialize the body detector with a pre-trained Faster R-CNN model.
        
        :param model_checkpoint_path: Path to the pre-trained model checkpoint.
        :param pipeline_config_path: Path to the pipeline config file.
        """
        # Load the pipeline config and build the detection model
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        model_config = configs['model']
        self.detection_model = model_builder.build(model_config=model_config, is_training=False)
        
        # Restore checkpoint
        ckpt = tf.compat.v2.train.Checkpoint(model=self.detection_model)
        ckpt.restore(model_checkpoint_path).expect_partial()
        
    def detect_body(self, img):
        """
        Detect body in the image using Faster R-CNN.
        
        :param img: Input image.
        :return: Bounding box in the format (left, top, right, bottom).
        """
        input_tensor = tf.convert_to_tensor(img)
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = self.detection_model(input_tensor)

        # Extract bounding box from detections
        boxes = detections['detection_boxes'][0].numpy()
        scores = detections['detection_scores'][0].numpy()
        classes = detections['detection_classes'][0].numpy()

        # Filter for 'person' class (class id = 1 in COCO) with high confidence
        body_boxes = boxes[(classes == 1) & (scores > 0.8)]

        if len(body_boxes) == 0:
            return None
        else:
            # Returning the box with highest score
            box = body_boxes[0]
            height, width, _ = img.shape
            left, top, right, bottom = box[1] * width, box[0] * height, box[3] * width, box[2] * height
            return (left, top, right, bottom)

# Initialize the body detector
body_detector = BodyDetector('path_to_model_checkpoint', 'path_to_pipeline_config')

# Update the process_single_face_image function to use body_detector.detect_body(image)
