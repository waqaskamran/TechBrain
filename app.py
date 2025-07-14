from flask import Flask, request, jsonify, render_template
from PIL import Image
import pytesseract
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
from flask import Flask, send_file
import os
from passporteye import read_mrz
import logging

from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
import easyocr
reader = easyocr.Reader(['en'], gpu=False)

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps,ImageEnhance
import pytesseract
import re
from flask import jsonify

from PIL import Image
import io
import socket
import os



import os
import yaml
from pathlib import Path



app = Flask(__name__)


# Disable all outbound internet connections (for debug only)




model_path = os.path.join('buffalo_l')
if not os.path.exists(model_path):
    raise RuntimeError(f"Buffalo model folder not found at {model_path}. Please download 'buffalo_l.zip' from InsightFace releases and unzip it.")
app_insight = FaceAnalysis(name='buffalo_l', root='./', providers=['CPUExecutionProvider'])
app_insight.prepare(ctx_id=0, det_size=(640, 640))
@app.route('/')
def home():
    return render_template('upload_form.html')


@app.route('/ocr', methods=['POST'])
def ocr_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        result = extract_text(file)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def clean_text(text):
    text = text.replace('\n', ' ')

    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')

    return text



@app.route('/ocr/easy', methods=['POST'])
def easy_ocr_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    try:
        image = Image.open(file.stream)
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        results = reader.readtext(open_cv_image, detail=0)
        text = " ".join(results)
        cleaned_text = clean_text_easy_ocr(text)
        return jsonify({'result': cleaned_text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def clean_text_easy_ocr(text):
    text = text.replace('\n', ' ')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    return text



@app.route('/extract_face', methods=['GET'])
def extract_face():
    images_folder = 'images'

    if not os.path.exists(images_folder):
        return "Error: 'images' folder not found", 404

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return "Error: No image found in 'images' folder", 404

    input_image = image_files[0]
    input_path = os.path.join(images_folder, input_image)

    img = cv2.imread(input_path)
    if img is None:
        return "Error: Failed to load image", 400

    faces = app_insight.get(img)

    if not faces:
        return "No faces detected in the image", 404

    face = faces[0]

    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    margin = 30
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(img.shape[1], x2 + margin)
    y2 = min(img.shape[0], y2 + margin)

    face_img = img[y1:y2, x1:x2]

    output_filename = 'face_' + input_image
    output_path = os.path.join(images_folder, output_filename)
    cv2.imwrite(output_path, face_img)

    return send_file(output_path, mimetype='image/jpeg')
@app.route('/detect_faces', methods=['GET'])
def detect_faces():

    images_folder = 'images'


    if not os.path.exists(images_folder):
        return "Error: 'images' folder not found", 404


    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return "Error: No image found in 'images' folder", 404


    input_image = image_files[0]
    input_path = os.path.join(images_folder, input_image)


    img = cv2.imread(input_path)
    if img is None:
        return "Error: Failed to load image", 400


    faces = app_insight.get(img)


    for face in faces:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)


    output_path = os.path.join(images_folder, 'output_' + input_image)
    cv2.imwrite(output_path, img)


    return send_file(output_path, mimetype='image/jpeg')

@app.route('/validate_profile_photo', methods=['GET'])
def validate_profile_photo():
    images_folder = 'images'

    if not os.path.exists(images_folder):
        return "Error: 'images' folder not found", 404

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return "Error: No image found in 'images' folder", 404

    input_image = image_files[0]
    input_path = os.path.join(images_folder, input_image)

    img = cv2.imread(input_path)
    if img is None:
        return "Error: Failed to load image", 400


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT)

    text_blocks = 0
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 50 and len(data['text'][i].strip()) > 5:
            text_blocks += 1
    if text_blocks > 2:
        return jsonify(valid=False, reason="Detected document-like structure with multiple text blocks")

    faces = app_insight.get(img)
    if len(faces) != 1:
        return jsonify(valid=False, reason="No face or multiple faces detected")

    face = faces[0]
    bbox = face.bbox.astype(int)
    face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    image_area = img.shape[0] * img.shape[1]
    face_ratio = face_area / image_area

    if face_ratio < 0.03:
        return jsonify(valid=False, reason="Face too small – possibly embedded in another image")

    return jsonify(valid=True)


@app.route('/compare_faces', methods=['GET'])
def compare_faces():

    images_folder = 'images'

    if not os.path.exists(images_folder):
        return jsonify({'error': 'images folder not found'}), 404

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) != 2:
        return jsonify({'error': 'Exactly 2 images are required in the images folder'}), 400

    img1_path = os.path.join(images_folder, image_files[0])
    img2_path = os.path.join(images_folder, image_files[1])

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        return jsonify({'error': 'Failed to load one or both images'}), 400

    faces1 = app_insight.get(img1)
    faces2 = app_insight.get(img2)

    if not faces1 or not faces2:
        return jsonify({'error': 'No faces detected in one or both images'}), 400

    embedding1 = faces1[0].normed_embedding
    embedding2 = faces2[0].normed_embedding

    similarity = np.dot(embedding1, embedding2)

    threshold = 0.4
    is_match =bool(similarity > threshold)

    return jsonify({
        'image1': image_files[0],
        'image2': image_files[1],
        'similarity_score': float(similarity),
        'is_match': is_match
    })

@app.route('/estimate_age_gender', methods=['GET'])
def estimate_age_gender():
    images_folder = 'images'

    if not os.path.exists(images_folder):
        return "Error: images folder not found", 404

    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        return "Error: No image found", 404

    input_image = image_files[0]
    input_path = os.path.join(images_folder, input_image)

    img = cv2.imread(input_path)
    if img is None:
        return "Error: Failed to load image", 400

    faces = app_insight.get(img)
    if not faces:
        return "Error: No faces detected", 400

    for face in faces:
        bbox = face.bbox.astype(int)
        gender = face['gender']
        age = face['age']

        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        label = f"{gender}, {age}"
        cv2.putText(img, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    output_path = os.path.join(images_folder, 'output_' + input_image)
    cv2.imwrite(output_path, img)

    return send_file(output_path, mimetype='image/jpeg')

@app.route('/extract_mrz', methods=['GET'])
def extract_mrz():
    try:
        images_folder = 'images'

        if not os.path.exists(images_folder):
            return jsonify({'error': 'images folder not found'}), 404

        image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not image_files:
            return jsonify({'error': 'No image found in images folder'}), 404

        input_image = image_files[0]
        input_path = os.path.join(images_folder, input_image)

        img = cv2.imread(input_path)
        if img is None:
            return jsonify({'error': 'Failed to load image'}), 400

        mrz = read_mrz(input_path)
        if mrz is None:
            mrz_result = {'error': 'No MRZ detected'}
            mrz_text = "No MRZ detected"
        else:
            mrz_result = mrz.to_dict()

            surname = mrz_result.get('surname', 'Unknown')
            givenname = mrz_result.get('givenname', 'Unknown')
            nationality = mrz_result.get('nationality', 'Unknown')
            passport_number = mrz_result.get('number', 'Unknown')

            mrz_fields = []
            if surname and surname != 'Unknown':
                mrz_fields.append(f"Surname: {surname}")
            if givenname and givenname != 'Unknown':
                mrz_fields.append(f"Given Name: {givenname}")
            if nationality and nationality != 'Unknown':
                mrz_fields.append(f"Nationality: {nationality}")
            if passport_number and passport_number != 'Unknown':
                mrz_fields.append(f"Passport: {passport_number}")

            mrz_text = ", ".join(mrz_fields) if mrz_fields else "Partial MRZ detected, some fields missing"

            cv2.putText(img, mrz_text, (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        output_path = os.path.join(images_folder, 'mrz_output_' + input_image)
        cv2.imwrite(output_path, img)

        return jsonify({
            'image': input_image,
            'mrz': mrz_result,
            'mrz_text': mrz_text,
            'annotated_image_path': output_path
        })

    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500



#driving mrz
def extract_text(file):
    preprocessed_img = preprocess_image(file)

    # Your original config but with English only to avoid Arabic interference
    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(preprocessed_img, config=custom_config, lang='eng')

    # Parse and format the extracted text
    return format_license_data(text)

def format_license_data(raw_text):
    """Extract and format license information"""
    lines = raw_text.split('\n')

    # Initialize variables
    license_no = ""
    name = ""
    nationality = ""
    date_of_birth = ""
    issue_date = ""
    expiry_date = ""
    place_of_issue = ""

    # Search for patterns in the text
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # License number - look for number pattern
        if 'License' in line and any(char.isdigit() for char in line):
            numbers = ''.join(filter(str.isdigit, line))
            if len(numbers) >= 6:
                license_no = numbers

        # Name - usually follows "Name" keyword
        if 'Name' in line or 'name' in line:
            # Extract everything after "Name" or similar
            parts = line.split()
            if len(parts) > 1:
                name_parts = []
                for part in parts[1:]:
                    if part.replace('.', '').replace(',', '').isalpha():
                        name_parts.append(part)
                if name_parts:
                    name = ' '.join(name_parts)

        # Dates - look for DD/MM/YYYY pattern
        import re
        dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', line)

        if dates:
            if 'Birth' in line or 'birth' in line:
                date_of_birth = dates[0]
            elif 'Issue' in line or 'issue' in line or 'tssuo' in line:
                issue_date = dates[0]
            elif 'Expiry' in line or 'expiry' in line or 'Expire' in line or 'expiy' in line:
                expiry_date = dates[0]

        # Nationality - extract value after the keyword
        if 'Nationality' in line or 'nationality' in line or 'Natlonallty' in line:
            # Split the line and find nationality value
            parts = line.split()
            nat_index = -1
            for i, part in enumerate(parts):
                if 'ationality' in part.lower():  # Matches Nationality, nationality, Natlonallty
                    nat_index = i
                    break

            if nat_index != -1 and nat_index + 1 < len(parts):
                # Get the word after nationality
                nationality = parts[nat_index + 1].strip('.,')

        # Place of issue - extract value after the keyword
        if 'Place' in line or 'place' in line:
            # Split the line and find place value
            parts = line.split()
            place_index = -1
            for i, part in enumerate(parts):
                if 'place' in part.lower():
                    place_index = i
                    break

            # Look for words after "Place of Issue" or similar
            if place_index != -1:
                for j in range(place_index + 1, len(parts)):
                    if parts[j].lower() not in ['of', 'issue', 'issuo']:
                        place_of_issue = parts[j].strip('.,')
                        break

    # If we couldn't parse properly, try a simpler approach



    # Fallback extraction for missing values
    if not nationality:
        # Look for any word that could be a country/nationality
        lines = raw_text.split('\n')
        for line in lines:
            words = line.split()
            for i, word in enumerate(words):
                if 'ationality' in word.lower() and i + 1 < len(words):
                    nationality = words[i + 1].strip('.,')
                    break
            if nationality:
                break

    if not place_of_issue:
        # Look for place after "Place" keyword
        lines = raw_text.split('\n')
        for line in lines:
            words = line.split()
            for i, word in enumerate(words):
                if 'place' in word.lower():
                    # Find the actual place name (skip "of", "issue" etc.)
                    for j in range(i + 1, len(words)):
                        if words[j].lower() not in ['of', 'issue', 'issuo', 'dubai']:
                            place_of_issue = words[j].strip('.,')
                            break
                    break
            if place_of_issue:
                break

    # Extract dates if not found
    if not date_of_birth or not issue_date or not expiry_date:
        import re
        all_dates = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', raw_text)
        if len(all_dates) >= 3:
            if not date_of_birth:
                date_of_birth = all_dates[0]
            if not issue_date:
                issue_date = all_dates[1]
            if not expiry_date:
                expiry_date = all_dates[2]

    # License number fallback
    if not license_no:
        import re
        numbers = re.findall(r'\d{6,8}', raw_text)
        if numbers:
            license_no = numbers[0]

    # Format the output
    formatted_output = []
    if license_no:
        formatted_output.append(f"* **License No**: {license_no}")
    if name:
        formatted_output.append(f"* **Name**: {name}")
    if nationality:
        formatted_output.append(f"* **Nationality**: {nationality}")
    if date_of_birth:
        formatted_output.append(f"* **Date of Birth**: {date_of_birth}")
    if issue_date:
        formatted_output.append(f"* **Issue Date**: {issue_date}")
    if expiry_date:
        formatted_output.append(f"* **Expiry Date**: {expiry_date}")
    if place_of_issue:
        formatted_output.append(f"* **Place of Issue**: {place_of_issue}")

    return '\n'.join(formatted_output) if formatted_output else raw_text

def preprocess_image(file_stream):
    # Your original preprocessing - keeping it exactly the same
    image_stream = np.asarray(bytearray(file_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)

    # Resize image (scale up for better OCR)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=30)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(gray, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return thresh


if __name__ == '__main__':
    app.run(debug=False)
