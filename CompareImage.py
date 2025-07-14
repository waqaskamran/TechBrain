import cv2
import numpy as np
from insightface.app import FaceAnalysis
from flask import Flask, send_file
import os

# Initialize Flask app
app_flask = Flask(__name__)

# Initialize InsightFace
app_insight = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app_insight.prepare(ctx_id=0, det_size=(640, 640))

# Define the route for face detection
@app_flask.route('/detect_faces', methods=['GET'])
def detect_faces():
    # Define the images folder
    images_folder = 'images'

    # Ensure the folder exists and get the first image
    if not os.path.exists(images_folder):
        return "Error: 'images' folder not found", 404

    # Get list of files in the images folder
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return "Error: No image found in 'images' folder", 404

    # Pick the first image (assuming only one image exists)
    input_image = image_files[0]
    input_path = os.path.join(images_folder, input_image)

    # Load the image
    img = cv2.imread(input_path)
    if img is None:
        return "Error: Failed to load image", 400

    # Detect faces
    faces = app_insight.get(img)

    # Draw bounding boxes on the image
    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Save the output image
    output_path = os.path.join(images_folder, 'output_' + input_image)
    cv2.imwrite(output_path, img)

    # Return the processed image
    return send_file(output_path, mimetype='image/jpeg')

# Run the Flask app
if __name__ == '__main__':
    # Ensure the 'images' folder exists
    if not os.path.exists('images'):
        os.makedirs('images')
    app_flask.run(debug=True, host='0.0.0.0', port=5000)