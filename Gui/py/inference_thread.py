#######################################################################################
# 추론 스레드 >> inference_model.test.py의 추론 코드를 비동기로 실행하는 워커 및 스레드
#######################################################################################
import sys
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal, QObject, pyqtSlot  # pyqtSlot: Qt 메타 객체 시스템에서 슬롯으로 인식하기 위해 필수
import time
import os

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
    
# 추론 모델 클래스 import
from inference_model.test import PCBDetector

# 추론 작업을 수행하는 워커 클래스 (QObject를 상속하여 시그널/슬롯 사용)
class InferenceWorker(QObject):
    finished = pyqtSignal(str)  # 추론 완료 시그널 (결과 디렉토리 경로 전달)
    log_msg = pyqtSignal(str)  # 로그 메시지 시그널
    error_occurred = pyqtSignal(str)  # 오류 발생 시그널
    model_loaded = pyqtSignal()  # 모델 로드 완료 시그널

    def __init__(self):
        super().__init__()
        self.detector = None  # PCBDetector 인스턴스
        self.current_weights_path = None  # 현재 로드된 가중치 파일 경로

    # 모델 가중치를 로드하는 함수 (Qt 메타 객체 시스템에서 슬롯으로 인식)
    @pyqtSlot(str)
    def load_model(self, weights_path):
        # 이미 같은 가중치가 로드되어 있으면 스킵
        if self.detector is not None and self.current_weights_path == str(weights_path):
            self.log_msg.emit(f"모델이 이미 로드되어 있습니다 (스킵): {os.path.basename(weights_path)}")
            self.model_loaded.emit()
            return
        try:
            self.log_msg.emit(f"모델 로딩 시작: {weights_path}")
            self.detector = PCBDetector(weights_path)  # PCBDetector 인스턴스 생성
            self.current_weights_path = str(weights_path)
            self.log_msg.emit("모델 로딩 완료!")
            self.model_loaded.emit()
        except Exception as e:
            self.error_occurred.emit(f"모델 로드 실패: {e}")
            import traceback
            traceback.print_exc()
            
    # 추론을 실행하는 함수 (Qt 메타 객체 시스템에서 슬롯으로 인식)
    @pyqtSlot(str, float, float)
    def run_inference(self, test_dir, conf, iou):
        # 모델이 로드되지 않았으면 오류 발생
        if not self.detector or not self.detector.model:
            self.error_occurred.emit("모델이 로드되지 않았습니다.")
            return

        try:
            self.log_msg.emit(f"추론 시작: {test_dir}")
            t_infer_start = time.perf_counter()
            # PCBDetector의 run_inference 메서드 호출
            out_dir = self.detector.run_inference(test_dir, conf=conf, iou=iou, show_mAP=True)
            t_infer_end = time.perf_counter()
            print(f"[Inference thread] 모델 infer: {t_infer_end - t_infer_start:.4f}s")
            self.finished.emit(out_dir)  # 추론 완료 시그널 발생
        except Exception as e:
            self.error_occurred.emit(f"추론 중 에러: {e}")
            import traceback
            traceback.print_exc()

# InferenceWorker를 실행하는 래퍼 스레드 클래스
class InferenceThread(QThread):
    """Worker를 가동하는 래퍼 스레드"""
    def __init__(self):
        super().__init__()
        self.worker = InferenceWorker()  # 워커 인스턴스 생성
        self.worker.moveToThread(self)  # 워커를 이 스레드로 이동 (비동기 실행을 위해)
    
    # 스레드 실행 함수 (이벤트 루프 시작)
    def run(self):
        self.exec_()  # Qt 이벤트 루프 실행