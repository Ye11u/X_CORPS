from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time

app = Flask(__name__)
camera = Picamera2()
# 해상도를 최대로 키움 
config = camera.create_preview_configuration(main={"size": (2048, 1536), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(2)

def crop_rotated_rectangle(image, box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
    return cv2.warpPerspective(image, M, (width, height))

def detect_and_crop_pcb(image):
    original = image.copy()
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    edges = cv2.Canny(blurred, 10, 90)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    largest = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cropped = crop_rotated_rectangle(original, box)
    return box, cropped

def process_image_orientation(image):
    if image is None: return None
    h, w = image.shape[:2]
    if h > w: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

@app.route('/frame')
def frame():
    frame_img = camera.capture_array()
    frame_display = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
    box, _ = detect_and_crop_pcb(frame_display)
    if box is not None:
        cv2.drawContours(frame_display, [box], -1, (0, 0, 255), 8)
    preview_small = cv2.resize(frame_display, (971, 601)) 

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode('.jpg', preview_small, encode_param)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/capture')
def capture():
    frame_img = camera.capture_array()
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)

    _, cropped = detect_and_crop_pcb(frame_img)

    if cropped is None:
        return Response("No PCB detected", status=404)

    final_image = process_image_orientation(cropped)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode('.jpg', final_image, encode_param)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

@app.route('/')
def index():

    return '''
    <html>
        <head>
            <title>Raspberry Pi PCB Detector</title>
            <meta http-equiv="refresh" content="600">
        </head>
        <body style="background-color: black; color: white; text-align: center;">
        <h2>Server is Running...</h2>
            <img src="/frame" width="971" height="601" style="border:2px solid #444;">
        </body>
    </html>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)