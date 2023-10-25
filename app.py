from flask import Flask, render_template
from sqlite_utility import SQLiteDB

import base64
from PIL import Image
from io import BytesIO

import matplotlib.pyplot as plt

from utility import deserialize_array
import io

app = Flask(__name__)

DB_PATH = "images_comparison.db"
db_util = SQLiteDB(DB_PATH)

def numpy_img_to_base64(img_np):
    img = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


@app.route('/test')
def test():
    all_data = db_util.fetch_all_image_features_with_roi()
    
    images_data = []
    
    for id, image_path, roi_blob, face_embedding, color_feature, texture_feature, pattern_feature in all_data:
        # Decode image_path if necessary
        if isinstance(image_path, bytes):
            image_path = image_path.decode('utf-8')
        
        # Convert roi_blob to bytes if it's a string
        if isinstance(roi_blob, str):
            roi_blob = bytes(roi_blob, 'utf-8')
        
        roi_np = deserialize_array(roi_blob)
        
        # Resize the image to thumbnail size
        img_pil = Image.fromarray(roi_np.astype('uint8'))
        img_pil = img_pil.resize((200, 200))
        buffer = BytesIO()
        img_pil.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

        face_embedding_array = deserialize_array(face_embedding)

        if face_embedding_array is not None:
            if face_embedding_array.ndim == 2:
                face_embedding_array = face_embedding_array[0, :2]
        else:
            print("face_embedding_array is None!")
        
        
        # Add all data for this image to the list
        images_data.append({
            'id': id,
            'image': img_str,
            'face_embedding': deserialize_array(face_embedding),
            'color_feature': deserialize_array(color_feature),
            'texture_feature': deserialize_array(texture_feature),
            'pattern_feature': deserialize_array(pattern_feature)
        })
    
    return render_template('test_image.html', images_data=images_data)


@app.route('/')
def index():
    
    return render_template('index.html', images='Hello')

if __name__ == "__main__":
    app.run(debug=True)
