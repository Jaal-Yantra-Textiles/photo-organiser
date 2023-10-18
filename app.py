from flask import Flask, render_template_string
import sqlite3
import numpy as np

app = Flask(__name__)

@app.route('/features')
def show_features():
    # Connect to the SQLite database
    conn = sqlite3.connect('image_features.db')
    cursor = conn.cursor()

    # Fetch all image features from the database
    cursor.execute("SELECT filename, face_features, dress_features FROM image_features")

    data = cursor.fetchall()

    # Convert binary data to a list of floats and take the first few values
    readable_data = [(filename, 
                      np.frombuffer(face_features, dtype=np.float32)[:5].tolist(), 
                      np.frombuffer(dress_features, dtype=np.float32)[:5].tolist()) 
                     for filename, face_features, dress_features in data]

    # Close the database connection
    conn.close()

    # Create a simple HTML template to display the features
    template = """
    <h1>Image Features</h1>
    <table border="1">
        <thead>
            <tr>
                <th>Filename</th>
                <th>Face Features (first 5 values)</th>
                <th>Dress Features (first 5 values)</th>
            </tr>
        </thead>
        <tbody>
            {% for row in data %}
            <tr>
                <td>{{ row[0] }}</td>
                <td>{{ row[1] }}</td>
                <td>{{ row[2] }}</td>
            </tr>
            {% endfor %}
        </tbody>
    </table>
    """

    return render_template_string(template, data=readable_data)

if __name__ == "__main__":
    app.run(debug=True)
