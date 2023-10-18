from PIL import Image
import os


def convert_to_jpeg(img_path):
    """Converts the image from .JPG to .JPEG format."""
    with Image.open(img_path) as img:
        new_path = img_path.replace('.JPG', '.JPEG')
        img.save(new_path, "JPEG", quality=100)
        os.remove(img_path)  # Remove the original .JPG file