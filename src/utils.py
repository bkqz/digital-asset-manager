import base64
import os

def encode_image(image_path):
    """
    Encodes an image to a base64 string for API consumption.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at: {image_path}")
        
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_file_name(image_path):
    """Returns the base filename from a path."""
    return os.path.basename(image_path)
