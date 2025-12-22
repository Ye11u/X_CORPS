#######################################################################################
# 훈련/(테스트, 실시간) 선택 화면: stack widget을 사용하여 모든 기능 페이지를 관리 
#######################################################################################
import sys
import res_rc
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from pathlib import Path

# 개발 환경(일반 폴더)과 배포 환경(임시 폴더 _MEIPASS) 모두에서 경로 오류를 방지하기 위한 함수 
if hasattr(sys, '_MEIPASS'):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from PCB_training_file_upload import PCB_Training_FileUpload # 훈련 파일 업로드 페이지 
from PCB_training_monitoring import PCB_Training_Monitoring # 훈련 모니터링 페이지 
from PCB_test_file_upload import PCB_Test_FileUpload # 테스트 파일 업로드 페이지 
from PCB_test_or_live import PCB_SelectTestOrLive # 테스트, 실시간 선택 페이지 
from PCB_live_streaming import PCB_LiveStreaming # 실시간 스트리밍 페이지 
from PCB_result_live_stream import PCB_Result_LiveStream # 실시간 스트리밍 결과 페이지  
from PCB_result import PCB_Result # 테스트 결과 페이지 
from inference_thread import InferenceThread # 추론 스레드 


# ui 파일 로드 
UI_FILE = ROOT / "ui" / "PCB_select_train_or_test.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))


class PCB_SelectTrainOrTest(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        self.infer_thread = InferenceThread() # 모델 추론을 위한 스레드 생성 (테스트, 실시간 테스트에 필요하므로 미리 로드)
        self.infer_thread.start()
        default_weights = ROOT / "inference_model/content/weights/last.pt" # 기본 모델 가중치 파일 경로 
        if default_weights.exists(): # 만약 모델 가중치 파일이 존재하면 
            from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
            QMetaObject.invokeMethod(self.infer_thread.worker, "load_model", 
                                     Qt.QueuedConnection, 
                                     Q_ARG(str, str(default_weights))) # 추론 시간 단축을 위해 미리 로드해둠 
        self._build_pages() # 페이지 초기화 (초기 화면)

        self.root_index = self.stackedWidget.currentIndex() # 현재 표시된 페이지의 인덱스 저장 (초기 화면)
        self.start_btn_2.clicked.connect(self.go_training_file_upload)  # Train 버튼 클릭 시 훈련 파일 업로드 페이지로 이동
        self.start_btn_3.clicked.connect(self.go_select_test_or_live_page)      # Test 버튼 클릭 시 (테스트 or 실시간) 선택 페이지로 이동
        
        if hasattr(self, "reset_btn_3"): # Reset 버튼 클릭 시 모든 페이지 리셋 (초기화면 == 현재 화면으로 돌아옴)
            self.reset_btn_3.clicked.connect(self.reset_all_pages)
            
    def _build_pages(self):
        self.training_file_upload_page = PCB_Training_FileUpload(parent=self) # 훈련 파일 업로드 페이지 생성
        self.training_monitoring_page  = PCB_Training_Monitoring(parent=self) # 훈련 모니터링 페이지 생성
        self.test_file_upload_page = PCB_Test_FileUpload(parent=self) # 테스트 파일 업로드 페이지 생성
        self.select_test_or_live_page = PCB_SelectTestOrLive(parent=self) # 테스트, 실시간 선택 페이지 생성 
        self.result_page               = PCB_Result(parent=self) # 테스트 결과 페이지 생성
        self.live_streaming_page       = PCB_LiveStreaming(parent=self, url="http://172.20.10.2:5000") # 실시간 스트리밍 페이지 생성, url 인자로는 라즈베리파이 서버 IP 주소를 넣어줘야함
        self.result_live_stream_page   = PCB_Result_LiveStream(parent=self) # 실시간 스트리밍 결과 페이지 생성

        self.test_file_upload_page.set_inference_worker(self.infer_thread.worker) # 테스트 파일 업로드 페이지에 추론 스레드 연결
        self.live_streaming_page.set_inference_worker(self.infer_thread.worker) # 실시간 스트리밍 페이지에 추론 스레드 연결
        
        # # stack widget에 생성한 모든 페이지 추가 
        self.stackedWidget.addWidget(self.training_file_upload_page) 
        self.stackedWidget.addWidget(self.training_monitoring_page) 
        self.stackedWidget.addWidget(self.test_file_upload_page)
        self.stackedWidget.addWidget(self.select_test_or_live_page)
        self.stackedWidget.addWidget(self.live_streaming_page)
        self.stackedWidget.addWidget(self.result_page)
        self.stackedWidget.addWidget(self.result_live_stream_page)

        self.training_file_upload_page.set_navigator( # 훈련 파일 업로드 페이지에 네비게이터 설정
            go_next=self.go_training_monitoring, # 훈련 파일 업로드의 다음 페이지: 훈련 모니터링 페이지
            go_back=self.go_root # 훈련 파일 업로드의 이전 페이지: 초기 화면
        )
        self.training_monitoring_page.set_navigator( # 훈련 모니터링 페이지에 네비게이터 설정
            go_test=self.go_test_file_upload, # 다음 페이지: 테스트 파일 업로드 페이지
            go_back=self.go_root # 이전 페이지: 초기 화면
        )
        self.select_test_or_live_page.set_navigator( # 테스트 or 실시간 선택 페이지에 네비게이터 설정
            go_live=self.go_live_streaming, # 다음 페이지 1: 실시간 스트리밍 페이지
            go_test_file=self.go_test_file_upload, # 다음 페이지 2: 테스트 파일 업로드 페이지
            go_back=self.go_root # 이전 페이지: 초기 화면
        )
        self.test_file_upload_page.set_navigator( # 테스트 파일 업로드 페이지에 네비게이터 설정
            go_next=self.go_result, # 다음: 결과 페이지 
            go_back=self.go_select_test_or_live_page # 이전: 테스트 or 실시간 선택 페이지 
        )
        self.result_page.set_navigator( # 테스트 결과 페이지 설정 
            go_back=self.go_root # 이전: 초기 화면 
        )
        self.live_streaming_page.set_navigator( # 실시간 스트리밍 페이지 설정 
            go_next=self.go_result_live_stream, # 다음: 실시간 스트리밍 결과 페이지 
            go_back=self.go_select_test_or_live_page # 이전: 테스트 or 실시간 선택 페이지 
        )
        self.result_live_stream_page.set_navigator( # 실시간 스트리밍 결과 페이지 설정 
            go_back=self.go_live_streaming # 이전: 실시간 스트리밍 페이지
        )

    def reset_all_pages(self): # 모든 페이지 리셋 (초기화면 == 현재 화면으로 돌아옴)
        print("전체 리셋 시작") 
        pages_to_reset = [ # 리셋할 페이지 리스트 (모든 페이지를 대상으로 함)
            getattr(self, "training_file_upload_page", None),
            getattr(self, "training_monitoring_page", None),
            getattr(self, "test_file_upload_page", None),
            getattr(self, "select_test_or_live_page", None),
            getattr(self, "live_streaming_page", None),
            getattr(self, "result_page", None),
        ]
        pages_to_reset = [p for p in pages_to_reset if p is not None]

        for page in pages_to_reset:
            try:
                if page is not None:
                    self.stackedWidget.removeWidget(page)
                    page.deleteLater()
            except Exception as e:
                print(f"페이지 제거 중 오류: {e}")
        self._build_pages()
        self.go_root() # 모든 페이지를 초기화시킨 뒤, 현재 페이지 == 초기화면으로 이동

    def go_training_file_upload(self): # 훈련 파일 업로드 화면으로 이동하는 함수 
        self.stackedWidget.setCurrentWidget(self.training_file_upload_page) # stack widget의 현재 페이지를 훈련 파일 업로드 페이지로 변경 

    # 같은 방법으로 다른 페이지들로 이동하는 함수도 구현 
    def go_training_monitoring(self): 
        self.stackedWidget.setCurrentWidget(self.training_monitoring_page)

    def go_test_file_upload(self):
        self.stackedWidget.setCurrentWidget(self.test_file_upload_page)

    def go_select_test_or_live_page(self):
        self.stackedWidget.setCurrentWidget(self.select_test_or_live_page)

    def go_result(self, start_time=None):
        self.stackedWidget.setCurrentWidget(self.result_page)
        if start_time is not None:
            if hasattr(self.result_page, 'set_start_time'):
                self.result_page.set_start_time(start_time)
                
    def go_result_live_stream(self, result_dir=None):
        if result_dir:
            if hasattr(self.result_live_stream_page, "load_results"):
                self.result_live_stream_page.load_results(result_dir)
            else:
                print("[Main] Error: pcb_live_result page has no 'load_results' method.")
        
        self.stackedWidget.setCurrentWidget(self.result_live_stream_page)

    def go_root(self): # 초기화면 == 현재 화면으로 이동하는 함수 
        self.stackedWidget.setCurrentIndex(self.root_index) 
    
    def go_live_streaming(self):
        self.stackedWidget.setCurrentWidget(self.live_streaming_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCB_SelectTrainOrTest()
    win.show()
    sys.exit(app.exec_())
