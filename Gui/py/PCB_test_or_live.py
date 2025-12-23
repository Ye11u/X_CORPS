#######################################################################################
# 테스트 or 실시간 처리 선택 페이지 
#######################################################################################
import sys, os
from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow, QApplication
from pathlib import Path

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

import res_rc
UI_FILE = ROOT / "ui" / "PCB_test_or_live.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))


class PCB_SelectTestOrLive(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        
        self._navigator = { # 네비게이터 설정 초기화 
            "go_live": None,
            "go_test_file": None,
            "go_back": None
        }

        # 버튼과 페이지 이동 함수 연결  
        self.start_btn_2.clicked.connect(self.go_live_page)   
        self.start_btn_3.clicked.connect(self.go_test_file_page)  
        if hasattr(self, "back_btn"): self.back_btn.clicked.connect(self.go_back_page)
    
   
    def set_navigator(self, go_live=None, go_test_file=None, go_back=None):
        self._navigator["go_live"] = go_live 
        self._navigator["go_test_file"] = go_test_file
        self._navigator["go_back"] = go_back

    def go_back_page(self): # 뒤로 가기 페이지에 초기 화면 (train/test 분기 화면)의 설정 연결 
        if self._navigator["go_back"]: self._navigator["go_back"]()

    def go_live_page(self): # 같은 방법으로 실시간 처리 페이지 연결 
        if self._navigator["go_live"]: self._navigator["go_live"]()

    def go_test_file_page(self): # 같은 방법으로 테스트 파일 업로드 페이지 연결 
        if self._navigator["go_test_file"]: self._navigator["go_test_file"]()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = PCB_SelectTestOrLive()
    win.show()
    sys.exit(app.exec_())
