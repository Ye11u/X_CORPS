import sys
import os
import pathlib
from pathlib import Path

from PyQt5 import uic
from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox, QLineEdit
from PyQt5.QtGui import QFont

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)

import res_rc

current_file_path = Path(__file__).resolve()
py_dir = current_file_path.parent 
project_root = current_file_path.parents[1]  

if str(py_dir) not in sys.path:
    sys.path.append(str(py_dir))
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from PCB_select_train_or_test import PCB_SelectTrainOrTest
except ImportError:
    print("모듈을 찾을 수 없습니다:", sys.path)

ui_file_path = resource_path(os.path.join("ui", "login.ui"))

FormClass, BaseClass = uic.loadUiType(ui_file_path)

user_info = {
    "1":    {"password": "1"},
    "inha0000":  {"password": "00000000"},
    "Tube": {"password": "qwer"},
    "Ryan": {"password": "zxcv"},
    "Muzi": {"password": "asdf"},
}

class LoginClass(BaseClass, FormClass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.settings = QSettings("PCB-Project", "PCB-GUI")

        if hasattr(self, "password_edit") and isinstance(self.password_edit, QLineEdit):
            self.password_edit.setEchoMode(QLineEdit.Password)
            self.password_edit.returnPressed.connect(self.try_login)

        if hasattr(self, "id_edit") and isinstance(self.id_edit, QLineEdit):
            self.id_edit.returnPressed.connect(self.try_login)
            saved_id = self.settings.value("last_id", "", type=str)
            if saved_id:
                self.id_edit.setText(saved_id)

        if hasattr(self, "login_btn"):
            self.login_btn.clicked.connect(self.try_login)

        if hasattr(self, "show_pw_btn") and hasattr(self, "password_edit"):
            try:
                self.show_pw_btn.setCheckable(True)
                self.show_pw_btn.toggled.connect(
                    lambda on: self.password_edit.setEchoMode(
                        QLineEdit.Normal if on else QLineEdit.Password
                    )
                )
            except Exception:
                pass

    def try_login(self):
        user_id = self.id_edit.text().strip() if hasattr(self, "id_edit") else ""
        password = self.password_edit.text().strip() if hasattr(self, "password_edit") else ""

        if not user_id or not password:
            QMessageBox.warning(self, "Error", "ID와 Password를 모두 입력하세요.")
            return

        if user_id in user_info and user_info[user_id]["password"] == password:
            if hasattr(self, "remember_check") and self.remember_check.isChecked():
                self.settings.setValue("last_id", user_id)
            else:
                self.settings.remove("last_id")

            self.next_win = PCB_SelectTrainOrTest()
            self.next_win.show()
            self.close()
        else:
            QMessageBox.critical(self, "Error", "ID 또는 Password가 올바르지 않습니다.")

if __name__ == "__main__":
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, False)
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    
    win = LoginClass()
    win.show()
    sys.exit(app.exec_())