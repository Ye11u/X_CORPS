import sys
import requests
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer

class CameraClient(QWidget):
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.init_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(100)

    def init_ui(self):
        self.video_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

    def update_frame(self):
        try:
            img_resp = requests.get(self.url, stream=True, timeout=0.5)
            bytes_data = b''
            for chunk in img_resp.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')
                b = bytes_data.find(b'\xff\xd9')
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, ch = rgb_image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qt_image))
                    break
        except Exception as e:
            print("Connection error:", e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    client = CameraClient("http://<라즈베리파이_IP>:5000/video_feed")
    client.show()
    sys.exit(app.exec_())
