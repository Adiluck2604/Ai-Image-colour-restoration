import os
import numpy as np
import cv2
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt

# Path of our caffemodel, prototxt, and numpy files
PROTOTXT = "model/colorization_deploy_v2.prototxt"
CAFFE_MODEL = "model/colorization_release_v2.caffemodel"
PTS_NPY = "model/pts_in_hull.npy"

# Create uploads folder
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model
net = cv2.dnn.readNetFromCaffe(PROTOTXT, CAFFE_MODEL)
pts = np.load(PTS_NPY)

layer1 = net.getLayerId("class8_ab")
layer2 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(layer1).blobs = [pts.astype("float32")]
net.getLayer(layer2).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return redirect(url_for('predict', filename=filename))
    return redirect(request.url)

@app.route('/predict/<filename>')
def predict(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    original_image = cv2.imread(filepath)

    # Convert image into gray scale and then to RGB
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    gray_rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

    # Normalize and convert to LAB
    normalized = gray_rgb_image.astype("float32") / 255.0
    lab_image = cv2.cvtColor(normalized, cv2.COLOR_RGB2LAB)
    resized = cv2.resize(lab_image, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict 'a' and 'b' channels
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (gray_rgb_image.shape[1], gray_rgb_image.shape[0]))

    # Combine L, a, b channels
    L = cv2.split(lab_image)[0]
    LAB_colored = np.concatenate((L[:, :, np.newaxis], ab), axis=2)

    # Convert LAB to RGB
    RGB_colored = cv2.cvtColor(LAB_colored, cv2.COLOR_LAB2RGB)
    RGB_colored = np.clip(RGB_colored, 0, 1)
    RGB_colored_255 = (255 * RGB_colored).astype("uint8")

    # Save processed images for display
    lab_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"lab_{filename}.png")
    colored_image_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"colored_{filename}.png")

    # Save LAB image
    LAB_colored_255 = (255 * ((LAB_colored + [0, 128, 128]) / [100, 255, 255])).astype("uint8")
    cv2.imwrite(lab_image_filepath, cv2.cvtColor(LAB_colored_255, cv2.COLOR_LAB2BGR))

    # Save colored image
    cv2.imwrite(colored_image_filepath, cv2.cvtColor(RGB_colored_255, cv2.COLOR_RGB2BGR))

    return render_template('result.html', input_image=url_for('static', filename=f"uploads/{filename}"),
                           lab_image=url_for('static', filename=f"uploads/lab_{filename}.png"),
                           output_image=url_for('static', filename=f"uploads/colored_{filename}.png"))

if __name__ == "__main__":
    app.run()
