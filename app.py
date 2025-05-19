from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import numpy as np
import base64
import io
import cv2
import os

app = Flask(__name__)
CORS(app)

def estimate_time_of_day_from_image(avg_brightness):
    if avg_brightness < 70:
        return "ช่วงกลางคืน"
    elif avg_brightness < 130:
        return "ช่วงเย็น"
    elif avg_brightness < 200:
        return "ช่วงบ่าย"
    else:
        return "ช่วงเช้า"

def analyze_cloud_and_rain(image_array):
    image = cv2.resize(image_array, (300, 300))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_cloud = np.array([0, 0, 180])
    upper_cloud = np.array([180, 50, 255])
    cloud_mask = cv2.inRange(hsv, lower_cloud, upper_cloud)

    cloud_pixels = cv2.countNonZero(cloud_mask)
    total_pixels = image.shape[0] * image.shape[1]
    cloud_percent = (cloud_pixels / total_pixels) * 100

    if cloud_percent > 30:
        rain_chance = 50
    elif cloud_percent > 20:
        rain_chance = 40
    elif cloud_percent > 15:
        rain_chance = 30
    elif cloud_percent > 10:
        rain_chance = 10
    else:
        rain_chance = 5

    return round(cloud_percent, 2), rain_chance

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'result': 'No image uploaded'}), 400

    file = request.files['image']
    image_pil = Image.open(file.stream).convert('RGB')
    image_array = np.array(image_pil)

    image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

    avg_brightness = np.mean(image_array)
    if avg_brightness < 100:
        weather = "ฝนตกหรือครึ้มฟ้า"
    elif avg_brightness < 180:
        weather = "อากาศปกติ"
    else:
        weather = "แดดจัดหรือท้องฟ้าโปร่ง"

    time_period = estimate_time_of_day_from_image(avg_brightness)
    cloud_percent, rain_chance = analyze_cloud_and_rain(image_cv2)

    result = f"{weather} ({time_period})\nเมฆปกคลุม: {cloud_percent}%\nโอกาสฝนตก: {rain_chance}%"

    buffered = io.BytesIO()
    image_pil.save(buffered, format="PNG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return jsonify({'result': result, 'image': encoded_image})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)