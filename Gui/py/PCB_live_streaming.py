#######################################################################################
# 라이브 스트리밍 화면 >> 서버로부터 영상 수신, 이미지 캡처, 8x4 분할 저장, 추론 실행
#######################################################################################
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

# 개발 환경(일반 폴더)과 배포 환경(임시 폴더 _MEIPASS) 모두에서 경로 오류를 방지하기 위한 함수 
if hasattr(sys, '_MEIPASS'):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

# 모듈 import 경로 문제 해결을 위해 시스템 경로에 루트 추가
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def resource_path(relative_path):
    """리소스 파일 경로를 반환하는 함수 (개발/배포 환경 모두 지원)"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
# UI 파일 로드 
UI_FILE = ROOT / "ui" / "PCB_live_streaming.ui"
IM_ROOT = ROOT / "inference_model"  # 추론 모델 루트 디렉토리
LIVE_TEST_DIR = IM_ROOT / "content" / "pcb_std" / "images" / "test_live_streaming"  # 라이브 스트리밍 테스트 이미지 저장 디렉토리

FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

class PCB_LiveStreaming(BaseClass, FormClass):
    def __init__(self, url=None, parent=None):
        super().__init__(parent)
        self.url = url  # 서버 URL (영상 스트리밍 및 캡처 요청용)
        self.setupUi(self)

        # 영상 프레임 업데이트를 위한 타이머 설정 (100ms 간격)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

        self._navigator = {"go_next": None, "go_back": None}  # 화면 전환을 위한 네비게이터 (초기 화면에서 함수를 주입받음)
        self.worker = None  # 추론 작업을 수행하는 워커 객체

        # 버튼 클릭 이벤트 연결 
        if hasattr(self, "back_btn"):
            self.back_btn.clicked.connect(self.go_back_page)
        
        if hasattr(self, "test_btn"):
            self.test_btn.clicked.connect(self.run_real_inference)
        elif hasattr(self, "finish_btn"):
             self.finish_btn.clicked.connect(self.run_real_inference)

    # 화면이 표시될 때 영상 스트리밍 타이머를 시작하는 함수
    def showEvent(self, event):
        super().showEvent(event)
        if not self.timer.isActive():
            self.timer.start(100)  # 100ms마다 프레임 업데이트
            print("[LIVE] Page Shown -> Timer Started")

    # 화면이 숨겨질 때 영상 스트리밍 타이머를 중지하는 함수
    def hideEvent(self, event):
        super().hideEvent(event)
        if self.timer.isActive():
            self.timer.stop()
            print("[LIVE] Page Hidden -> Timer Stopped")

    # 초기 화면에서 추론 워커를 받아오는 함수 
    def set_inference_worker(self, worker):
        self.worker = worker
        self.worker.finished.connect(self._on_inference_finished)  # 추론 완료 시 호출될 함수 연결

    # 서버로부터 최신 프레임을 받아와서 화면에 표시하는 함수
    def update_frame(self):
        if not self.url: return 
        try:
            # 서버에서 프레임 요청
            img_resp = requests.get(self.url + "/frame", timeout=0.5)
            if img_resp.status_code != 200: return

            # 응답 데이터를 이미지로 디코딩
            img_array = np.frombuffer(img_resp.content, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None: return
            
            # BGR을 RGB로 변환 (OpenCV는 BGR, Qt는 RGB 사용)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))
        except Exception:
            pass

    # 초기 화면에서 페이지 이동 함수를 받아오는 함수 
    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next
        self._navigator["go_back"] = go_back

    # 뒤로 가기 함수 
    def go_back_page(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    # 서버에 이미지 캡처를 요청하고, 받은 이미지를 8x4 그리드로 분할하여 저장하는 함수
    def capture_and_save_images(self):
        print("[LIVE] Requesting capture to server...")
        try:
            # 서버에 캡처 요청
            resp = requests.get(self.url + "/capture", timeout=5)
            
            if resp.status_code != 200:
                QMessageBox.warning(self, "통신 오류", f"이미지 캡처 실패: {resp.text}")
                return None

            # 응답 데이터를 이미지로 디코딩
            img_array = np.frombuffer(resp.content, np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if image is None:
                QMessageBox.warning(self, "오류", "받은 이미지가 비어있습니다.")
                return None
            
            # 세로가 가로보다 길면 90도 회전 (가로 방향으로 변환)
            h, w = image.shape[:2]
            if h > w:
                print("[LIVE] Creating landscape image (Rotating 90 deg)")
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                h, w = image.shape[:2]
            
            # 타임스탬프로 저장 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = LIVE_TEST_DIR / timestamp
            
            if not save_dir.exists():
                os.makedirs(save_dir, exist_ok=True)

            print(f"[LIVE] Saving images to: {save_dir}")
            # 원본 이미지 저장
            original_filename = f"{timestamp}_original.jpg"
            cv2.imwrite(str(save_dir / original_filename), image)
            print(f"[LIVE] Original full image saved: {original_filename}")

            # 이미지를 8x4 그리드로 분할 (4행 8열)
            dy = h // 4  # 행당 높이
            dx = w // 8  # 열당 너비
            
            count = 0
            for r in range(4):  # 4행
                for c in range(8):  # 8열
                    y_start = r * dy
                    y_end = (r + 1) * dy
                    x_start = c * dx
                    x_end = (c + 1) * dx

                    # 마지막 행/열은 남은 픽셀까지 포함
                    if r == 3: y_end = h
                    if c == 7: x_end = w

                    # 이미지 패치 추출
                    patch = image[y_start:y_end, x_start:x_end]
                    
                    # 파일명 생성 (00~31)
                    filename = f"{timestamp}_{count:02d}.jpg"
                    save_path = save_dir / filename
                    
                    cv2.imwrite(str(save_path), patch)
                    count += 1
            
            return save_dir

        except Exception as e:
            QMessageBox.critical(self, "캡처 오류", f"이미지 처리 중 오류 발생:\n{e}")
            return None

    # 이미지 캡처 및 분할 저장 후 추론을 실행하는 함수
    def run_real_inference(self):
        if not self.worker:
            QMessageBox.critical(self, "오류", "추론 시스템이 연결되지 않았습니다.")
            return
        
        # 이미지 캡처 및 분할 저장
        target_dir = self.capture_and_save_images()
        
        if target_dir is None:
            # 캡처 실패 시 타이머 재시작
            if not self.timer.isActive():
                self.timer.start(100)
            return

        # 추론 중에는 버튼 비활성화 및 대기 커서 표시
        if hasattr(self, "test_btn"): self.test_btn.setEnabled(False)
        if hasattr(self, "finish_btn"): self.finish_btn.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        
        # 워커에 추론 실행 요청 (비동기)
        QMetaObject.invokeMethod(self.worker, "run_inference",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, str(target_dir)),  # 이미지 디렉토리 경로
                                 Q_ARG(float, 0.25),  # confidence 임계값
                                 Q_ARG(float, 0.45))  # IoU 임계값

    # 추론 완료 시 호출되는 콜백 함수
    def _on_inference_finished(self, result_dir):
        QApplication.restoreOverrideCursor()  # 커서 복원
        if hasattr(self, "test_btn"): self.test_btn.setEnabled(True)  # 버튼 다시 활성화
        if hasattr(self, "finish_btn"): self.finish_btn.setEnabled(True)
        
        print(f"[LIVE] Inference Done. Result path: {result_dir}")

        # 다음 페이지로 이동 (결과 디렉토리 경로 전달)
        if self._navigator["go_next"]:
            self._navigator["go_next"](result_dir=result_dir)

# 테스트용 메인 함수
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCB_LiveStreaming(url="http://172.20.10.2:5000")  # 서버 URL 설정, 서버에 뜨는 로그값으로 바꿔줘야함!!!!!!!!!!!
    win.show()
    sys.exit(app.exec_())