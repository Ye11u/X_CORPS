import sys, os, shutil, time
import requests
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
from PyQt5.QtCore import QTimer, QMetaObject, Q_ARG, Qt
from PyQt5.QtGui import QImage, QPixmap

if hasattr(sys, '_MEIPASS'):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
UI_FILE = ROOT / "ui" / "PCB_live_streaming.ui"
IM_ROOT = ROOT / "inference_model"
LIVE_TEST_DIR = IM_ROOT / "content" / "pcb_std" / "images" / "test_live_streaming"

FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

class PCB_LiveStreaming(BaseClass, FormClass):
    def __init__(self, url=None, parent=None):
        super().__init__(parent)
        self.url = url
        self.setupUi(self)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._navigator = {"go_next": None, "go_back": None}
        self.worker = None  

        if hasattr(self, "back_btn"):
            self.back_btn.clicked.connect(self.go_back_page)
        
        if hasattr(self, "test_btn"):
            self.test_btn.clicked.connect(self.run_real_inference)
        elif hasattr(self, "finish_btn"):
             self.finish_btn.clicked.connect(self.run_real_inference)

    def showEvent(self, event):
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start(100)
            print("[LIVE] Page Shown -> Timer Started")

    def hideEvent(self, event):
        super().hideEvent(event)
        if self.timer.isActive():
            self.timer.stop()
            print("[LIVE] Page Hidden -> Timer Stopped")

    def set_inference_worker(self, worker):
        self.worker = worker
        self.worker.finished.connect(self._on_inference_finished)

    def update_frame(self):
        if not self.url: return 
        try:
            img_resp = requests.get(self.url + "/frame", timeout=0.5)
            if img_resp.status_code != 200: return

            img_array = np.frombuffer(img_resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None: return
            
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception:
            pass

    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next
        self._navigator["go_back"] = go_back

    def go_back_page(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    def capture_and_save_images(self):
        print("[LIVE] Requesting capture to server...")
        try:
            resp = requests.get(self.url + "/capture", timeout=5)
            
            if resp.status_code != 200:
                QMessageBox.warning(self, "통신 오류", f"이미지 캡처 실패: {resp.text}")
                return None

            img_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                QMessageBox.warning(self, "오류", "받은 이미지가 비어있습니다.")
                return None
            h, w = image.shape[:2]
            if h > w:
                print("[LIVE] Creating landscape image (Rotating 90 deg)")
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                h, w = image.shape[:2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = LIVE_TEST_DIR / timestamp
            
            if not save_dir.exists():
                os.makedirs(save_dir, exist_ok=True)

            print(f"[LIVE] Saving images to: {save_dir}")
            original_filename = f"{timestamp}_original.jpg"
            cv2.imwrite(str(save_dir / original_filename), image)
            print(f"[LIVE] Original full image saved: {original_filename}")

            dy = h // 4
            dx = w // 8
            
            count = 0
            for r in range(4): 
                for c in range(8): 
                    y_start = r * dy
                    y_end = (r + 1) * dy
                    x_start = c * dx
                    x_end = (c + 1) * dx

                    if r == 3: y_end = h
                    if c == 7: x_end = w

                    patch = image[y_start:y_end, x_start:x_end]
                    
                    filename = f"{timestamp}_{count:02d}.jpg"
                    save_path = save_dir / filename
                    
                    cv2.imwrite(str(save_path), patch)
                    count += 1
            
            return save_dir

        except Exception as e:
            QMessageBox.critical(self, "캡처 오류", f"이미지 처리 중 오류 발생:\n{e}")
            return None

    def run_real_inference(self):
        if not self.worker:
            QMessageBox.critical(self, "오류", "추론 시스템이 연결되지 않았습니다.")
            return
        target_dir = self.capture_and_save_images()
        
        if target_dir is None:
            if not self.timer.isActive():
                self.timer.start(100)
            return

        if hasattr(self, "test_btn"): self.test_btn.setEnabled(False)
        if hasattr(self, "finish_btn"): self.finish_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QMetaObject.invokeMethod(self.worker, "run_inference",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, str(target_dir)),
                                 Q_ARG(float, 0.25), # conf
                                 Q_ARG(float, 0.45)) # iou

    def _on_inference_finished(self, result_dir):
        QApplication.restoreOverrideCursor()
        if hasattr(self, "test_btn"): self.test_btn.setEnabled(True)
        if hasattr(self, "finish_btn"): self.finish_btn.setEnabled(True)
        
        print(f"[LIVE] Inference Done. Result path: {result_dir}")

        if self._navigator["go_next"]:
            self._navigator["go_next"](result_dir=result_dir)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCB_LiveStreaming(url="http://172.20.10.2:5000") 
    win.show()
    sys.exit(app.exec_())