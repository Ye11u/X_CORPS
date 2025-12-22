import sys
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot  # <--- pyqtSlot 추가 필수!
import time
import os

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
    
from inference_model.test import PCBDetector

class InferenceWorker(QObject):
    finished = pyqtSignal(str)
    log_msg = pyqtSignal(str)      
    error_occurred = pyqtSignal(str) 
    model_loaded = pyqtSignal() 

    def __init__(self):
        super().__init__()
        self.detector = None
        self.current_weights_path = None

    @pyqtSlot(str)
    def load_model(self, weights_path):
        if self.detector is not None and self.current_weights_path == str(weights_path):
            self.log_msg.emit(f"모델이 이미 로드되어 있습니다 (스킵): {os.path.basename(weights_path)}")
            self.model_loaded.emit()
            return
        try:
            self.log_msg.emit(f"모델 로딩 시작: {weights_path}")
            self.detector = PCBDetector(weights_path)
            self.current_weights_path = str(weights_path)
            self.log_msg.emit("모델 로딩 완료!")
            self.model_loaded.emit()
        except Exception as e:
            self.error_occurred.emit(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            
    @pyqtSlot(str, float, float)
    def run_inference(self, test_dir, conf, iou):
        if not self.detector or not self.detector.model:
            self.error_occurred.emit("모델이 로드되지 않았습니다.")
            return

        try:
            self.log_msg.emit(f"추론 시작: {test_dir}")
            t_infer_start = time.perf_counter()
            out_dir = self.detector.run_inference(test_dir, conf=conf, iou=iou, show_mAP=True)
            t_infer_end = time.perf_counter()
            print(f"[Inference thread] 모델 infer: {t_infer_end - t_infer_start:.4f}s")
            self.finished.emit(out_dir)
        except Exception as e:
            self.error_occurred.emit(f"추론 중 에러: {e}")
            import traceback
            traceback.print_exc()

class InferenceThread(QThread):
    """Worker를 가동하는 래퍼 스레드"""
    def __init__(self):
        super().__init__()
        self.worker = InferenceWorker()
        self.worker.moveToThread(self)
    
    def run(self):
        self.exec_()