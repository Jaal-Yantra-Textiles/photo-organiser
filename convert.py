from PIL import Image
import logging
import os
from logger_util import logger

def convert_to_jpeg(img_file, output_dir,resize, output_width=636, output_height=1024):
    logger.info(f"Processing images {img_file} to jpeg with resize as {resize}")
    try:
        with Image.open(img_file) as im:
            # Resize the image if the resize flag is True
            if resize:
                im = im.resize((output_width, output_height))
            
            # Create output directory if it doesn't exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Convert .JPG to .JPEG and save to specified directory
            output_file_name = os.path.basename(img_file).replace('.JPG', '.jpeg')
            output_path = os.path.join(output_dir, output_file_name)
            
            im.save(output_path, 'jpeg')
            return output_path
    except Exception as e:
        logger.error(f"Error converting {img_file} to jpeg. Error: {e}")
        return None