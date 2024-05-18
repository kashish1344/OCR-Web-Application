
# 📄 OCR Web Application

This repository contains a web application for **Optical Character Recognition (OCR)** using Flask, OpenCV, and EasyOCR. The application allows users to upload images, processes these images to correct orientation and skew, and extracts text using OCR technology. The extracted text is then presented to the user.

## ✨ Features

- **📤 Image Upload:** Users can upload images through the web interface.
- **🔧 Image Preprocessing:** Automatically corrects image orientation and skew to improve OCR accuracy.
- **📝 Text Extraction:** Utilizes EasyOCR to read and extract text from images.
- **📑 Text Grouping:** Groups text segments into coherent lines for better readability.
- **📋 API Response:** Returns the extracted text as a JSON response.

## 🛠 Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x
- Flask
- OpenCV
- EasyOCR
- PIL (Pillow)

## 📥 Installation

1. **Clone this repository:**

    ```bash
    git clone https://github.com/your-username/ocr-web-app.git
    cd ocr-web-app
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Ensure the `uploads` directory exists:**

    ```bash
    mkdir -p uploads
    ```

## 🚀 Usage

1. **Run the Flask application:**

    ```bash
    python app.py
    ```

2. **Open your web browser and go to `http://127.0.0.1:5000`.**

3. **Upload an image file through the web interface.**

4. **The application will process the image, extract the text, and display it on the web page.**

## 🧩 Code Overview

### `app.py`

The main application file contains the following key functions:

- **Image Preprocessing:**
  - `correct_skew(image)`: Corrects skew in the image using Hough Line Transform.
  - `correct_orientation(img, image_path)`: Corrects image orientation using EXIF data.
  - `preprocess_image(image_path)`: Combines the above methods and applies additional preprocessing steps.
  
- **Text Extraction:**
  - `read_text_from_image(image_path)`: Uses EasyOCR to read text from the preprocessed image.
  
- **Text Grouping:**
  - `are_on_same_line(box1, box2, dynamic_threshold=True)`: Determines if two text boxes are on the same line.
  - `group_text_segments_into_lines(results)`: Groups text segments into coherent lines.

- **Flask Routes:**
  - `/`: Serves the main HTML page.
  - `/upload`: Handles image upload and text extraction, returning the result as JSON.

### Templates

- **`templates/index.html`**: The main HTML page for the web interface.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for more information.

---

Happy coding! If you have any questions or need further assistance, feel free to open an issue.
