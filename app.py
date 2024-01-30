from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET','POST'])
def detect_face():
    # Check if the request contains the 'file' field
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if the file is an allowed format (e.g., image/jpeg, image/png)
    if file.mimetype.split('/')[0] != 'image':
        return jsonify({'error': 'Unsupported file format'}), 400
    
    # Read the uploaded image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Perform face detection
    if img is None:
        return jsonify({'error': 'Unable to load image'}), 400
    
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    
    # Check if faces are detected
    if len(faces) > 0:
        result = {'message': 'Yes', 'num_faces': len(faces)}
    else:
        result = {'message': 'No', 'num_faces': 0}
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
