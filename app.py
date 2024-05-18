import os
import cv2
import numpy as np
import easyocr
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Ensure the uploads directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to correct skew in an image
def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    edged = cv2.Canny(blur, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    if lines is not None:
        angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for line in lines for x1, y1, x2, y2 in line]
        angle_deg = np.median(angles)
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return corrected
    return image

# Function to correct orientation using EXIF data
def correct_orientation(img, image_path):
    try:
        pil_img = Image.open(image_path)
        exif_data = pil_img._getexif()
        if not exif_data:
            return img
        orientation_tag = 274
        if orientation_tag in exif_data:
            orientation = exif_data[orientation_tag]
            if orientation == 3:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif orientation == 6:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        print(f"Failed to correct orientation: {e}")
    return img

# Function to preprocess image
def preprocess_image(image_path):
    if not os.path.isfile(image_path):
        raise FileNotFoundError('The specified image file does not exist.')

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError('The specified image could not be read.')

    image = correct_orientation(image, image_path)
    image = correct_skew(image)

    if image.dtype != np.float32:
        image = image.astype(np.float32)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 23, 23)

    if np.mean(adaptive_threshold) < 127:
        adaptive_threshold = cv2.bitwise_not(adaptive_threshold)

    resized = cv2.resize(adaptive_threshold, None, fx=1.9, fy=2, interpolation=cv2.INTER_CUBIC)
    denoised = cv2.fastNlMeansDenoising(resized, h=40)
    cv2.imwrite('processed_image.png', denoised, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    return denoised

# Function to read text from image using EasyOCR
def read_text_from_image(image_path):
    reader = easyocr.Reader(['en'])
    preprocessed_image = preprocess_image(image_path)
    results = reader.readtext(preprocessed_image, detail=1, paragraph=False)
    return results

# Function to check if two text boxes are on the same line
def are_on_same_line(box1, box2, dynamic_threshold=True):
    mid_y1 = (box1[0][1] + box1[2][1]) / 2
    mid_y2 = (box2[0][1] + box2[2][1]) / 2
    height1 = abs(box1[0][1] - box1[2][1])
    height2 = abs(box2[0][1] - box2[2][1])
    avg_height = (height1 + height2) / 2
    y_threshold = avg_height * 0.5 if dynamic_threshold else 17
    return abs(mid_y1 - mid_y2) < y_threshold

# Function to group text segments into lines
def group_text_segments_into_lines(results):
    results.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
    lines = []

    for box, text, _ in results:
        found_line = False
        for line in lines:
            if are_on_same_line(line['box'], box):
                line['text'].append((box, text))
                found_line = True
                break
        if not found_line:
            lines.append({'box': box, 'text': [(box, text)]})

    lines.sort(key=lambda x: x['text'][0][0][0][1])
    extracted_text = ""

    for line in lines:
        sorted_text_segments = sorted(line['text'], key=lambda x: x[0][0][0])
        line_texts = []

        for i, (box, text) in enumerate(sorted_text_segments):
            if i > 0:
                prev_box = sorted_text_segments[i - 1][0]
                curr_box = box
                height1 = abs(prev_box[0][1] - prev_box[2][1])
                height2 = abs(curr_box[0][1] - curr_box[2][1])
                avg_height = (height1 + height2) / 2
                gap = curr_box[0][0] - prev_box[2][0]
                if gap > avg_height * 0.1:
                    line_texts.append(" ")
            line_texts.append(text)

        line_text = "".join(line_texts)
        extracted_text += line_text + "\n"
    
    return extracted_text

# Route to serve the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Flask route to handle image upload and text extraction
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        results = read_text_from_image(file_path)
        extracted_text = group_text_segments_into_lines(results)
        return jsonify({'extracted_text': extracted_text}), 200
    except Exception as e:
        return jsonify({'error': "Please Try Again"}), 500
    
    finally:
        # Ensure the image file is removed after processing or in case of an error
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
