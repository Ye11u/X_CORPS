import sys
import res_rc
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from pathlib import Path

if hasattr(sys, '_MEIPASS'):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from PCB_training_file_upload import PCB_Training_FileUpload
from PCB_training_monitoring import PCB_Training_Monitoring
from PCB_test_file_upload import PCB_Test_FileUpload
from PCB_test_or_live import PCB_SelectTestOrLive
from PCB_live_streaming import PCB_LiveStreaming
from PCB_result_live_stream import PCB_Result_LiveStream
from PCB_result import PCB_Result
from inference_thread import InferenceThread

UI_FILE = ROOT / "ui" / "PCB_select_train_or_test.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))


class PCB_SelectTrainOrTest(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        self.infer_thread = InferenceThread()
        self.infer_thread.start()
        default_weights = ROOT / "inference_model/content/weights/last.pt"
        if default_weights.exists():
            from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
            QMetaObject.invokeMethod(self.infer_thread.worker, "load_model", 
                                     Qt.QueuedConnection, 
                                     Q_ARG(str, str(default_weights)))
        self._build_pages()

        self.root_index = self.stackedWidget.currentIndex()

        self.start_btn_2.clicked.connect(self.go_training_file_upload)  # Train
        self.start_btn_3.clicked.connect(self.go_select_test_or_live_page)      # Test
        
        if hasattr(self, "reset_btn_3"):
            self.reset_btn_3.clicked.connect(self.reset_all_pages)
            
    def _build_pages(self):
        self.training_file_upload_page = PCB_Training_FileUpload(parent=self)
        self.training_monitoring_page  = PCB_Training_Monitoring(parent=self)
        self.test_file_upload_page = PCB_Test_FileUpload(parent=self) 
        self.select_test_or_live_page = PCB_SelectTestOrLive(parent=self)
        self.result_page               = PCB_Result(parent=self)
        self.live_streaming_page       = PCB_LiveStreaming(parent=self, url="http://172.20.10.2:5000")
        self.result_live_stream_page   = PCB_Result_LiveStream(parent=self)

        self.test_file_upload_page.set_inference_worker(self.infer_thread.worker)
        self.live_streaming_page.set_inference_worker(self.infer_thread.worker)
        
        self.stackedWidget.addWidget(self.training_file_upload_page)
        self.stackedWidget.addWidget(self.training_monitoring_page) 
        self.stackedWidget.addWidget(self.test_file_upload_page)
        self.stackedWidget.addWidget(self.select_test_or_live_page)
        self.stackedWidget.addWidget(self.live_streaming_page)
        self.stackedWidget.addWidget(self.result_page)
        self.stackedWidget.addWidget(self.result_live_stream_page)

        self.training_file_upload_page.set_navigator(
            go_next=self.go_training_monitoring,
            go_back=self.go_root
        )
        self.training_monitoring_page.set_navigator(
            go_test=self.go_test_file_upload,
            go_back=self.go_root
        )
        self.select_test_or_live_page.set_navigator(
            go_live=self.go_live_streaming,
            go_test_file=self.go_test_file_upload,
            go_back=self.go_root 
        )
        self.test_file_upload_page.set_navigator(
            go_next=self.go_result,
            go_back=self.go_select_test_or_live_page
        )
        self.result_page.set_navigator(
            go_back=self.go_root
        )
        self.result_live_stream_page.set_navigator(
            go_back=self.go_live_streaming
        )
        self.live_streaming_page.set_navigator(
            go_next=self.go_result_live_stream,
            go_back=self.go_select_test_or_live_page 
        )

    def reset_all_pages(self):
        print("🔄 전체 리셋 시작")

        pages_to_reset = [
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
                print(f"⚠ 페이지 제거 중 오류: {e}")

        self._build_pages()
        self.go_root()

    def go_training_file_upload(self):
        self.stackedWidget.setCurrentWidget(self.training_file_upload_page)

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

    def go_root(self):
        self.stackedWidget.setCurrentIndex(self.root_index)
    
    def go_live_streaming(self):
        self.stackedWidget.setCurrentWidget(self.live_streaming_page)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCB_SelectTrainOrTest()
    win.show()
    sys.exit(app.exec_())
