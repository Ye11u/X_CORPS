import os, shutil, json, traceback, time, sys
import subprocess 
from pathlib import Path
from PyQt5 import uic
import res_rc
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from dragdrop_uploader import FileDropWidget
from PyQt5.QtCore import QEvent, Qt

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
    

UI_FILE = ROOT / "ui" / "PCB_training_file_upload.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

class PCB_Training_FileUpload(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._navigator = {"go_next": None, "go_back": None}

        self.dataset_path = None 
        self.model_py_path = None 

        self.file_btn.clicked.connect(self.pick_dataset_zip)
        self.file_btn_2.clicked.connect(self.pick_model_py)
        self.next_btn.clicked.connect(self.next_page)
        self.back_btn.clicked.connect(self.back_page)

        self._update_btn_texts()
        self.file_btn.setAcceptDrops(True) 
        self.file_btn_2.setAcceptDrops(True) 
        self.file_btn.installEventFilter(self)
        self.file_btn_2.installEventFilter(self)

        self._btn1_style0 = self.file_btn.styleSheet() or ""
        self._btn2_style0 = self.file_btn_2.styleSheet() or ""
        self._hover_style = """
        QPushButton, QToolButton {
            border: 2px dashed #1C8CFF; 
            background: #EAF6FF;     
            border-radius: 10px;
        }
        """

        self.dropper = FileDropWidget(
            text="여기로 .zip(데이터셋) 또는 .py(모델) 드래그",
            accept_exts=['.zip', '.py'],
            accept_dirs=False,  
            recursive=False,
            parent=self
        )
        self.dropper.filesDropped.connect(self._on_files_dropped)
        container = self.file_btn.parentWidget()
        target_layout = None
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLayout
        while container is not None and target_layout is None:
            lay = container.layout()
            if isinstance(lay, (QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout, QLayout)):
                target_layout = lay
                break
            container = container.parentWidget()

        if target_layout is None:
            cw = getattr(self, "centralwidget", None)
            if cw is not None and cw.layout() is not None:
                cw.layout().addWidget(self.dropper)
            else:
                from PyQt5.QtWidgets import QVBoxLayout
                root_layout = QVBoxLayout(self)
                self.setLayout(root_layout)
                root_layout.addWidget(self.dropper)
        else:
            target_layout.addWidget(self.dropper)



    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next
        self._navigator["go_back"] = go_back

    def pick_dataset_zip(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Dataset (.zip)", "", "Zip Files (*.zip)"
        )
        if path:
            self.dataset_path = path
            self._update_btn_texts()

    def pick_model_py(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Script (.py)", "", "Python Files (*.py)"
        )
        if path:
            self.model_py_path = path
            self._update_btn_texts()

    def _on_files_dropped(self, paths):
        if not paths:
            return

        picked_zip = None
        picked_py = None

        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            if ext == ".zip":
                picked_zip = p
            elif ext == ".py":
                picked_py = p

        changed = False
        if picked_zip:
            self.dataset_path = picked_zip
            changed = True
        if picked_py:
            self.model_py_path = picked_py
            changed = True

        if changed:
            self._update_btn_texts()
        else:
            QMessageBox.information(self, "안내", ".zip 또는 .py 파일만 드래그해 주세요.")


    def _update_btn_texts(self):
        self.file_btn.setText(
            os.path.basename(self.dataset_path) if self.dataset_path else "Select .zip dataset"
        )
        self.file_btn_2.setText(
            os.path.basename(self.model_py_path) if self.model_py_path else "Select model .py"
        )

    def eventFilter(self, obj, event):
        if obj is self.file_btn:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        p = url.toLocalFile()
                        if p and os.path.splitext(p)[1].lower() == ".zip":
                            obj.setStyleSheet(self._hover_style)
                            event.acceptProposedAction()
                            return True
                return False
            if event.type() == QEvent.DragLeave:
                obj.setStyleSheet(self._btn1_style0)      
                return False

            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn1_style0)
                if event.mimeData().hasUrls():
                    last_zip = None
                    for url in event.mimeData().urls():
                        p = url.toLocalFile()
                        if p and os.path.splitext(p)[1].lower() == ".zip":
                            last_zip = p
                    if last_zip:
                        self.dataset_path = last_zip
                        self._update_btn_texts()
                        event.acceptProposedAction()
                        return True
                    else:
                        QMessageBox.information(self, "안내", ".zip 파일만 드롭할 수 있습니다.")
                        return True
                return False

        if obj is self.file_btn_2:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        p = url.toLocalFile()
                        if p and os.path.splitext(p)[1].lower() == ".py":
                            obj.setStyleSheet(self._hover_style)
                            event.acceptProposedAction()
                            return True
                return False

            if event.type() == QEvent.DragLeave:
                obj.setStyleSheet(self._btn2_style0)  
                return False


            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn2_style0)  
                if event.mimeData().hasUrls():
                    last_py = None
                    for url in event.mimeData().urls():
                        p = url.toLocalFile()
                        if p and os.path.splitext(p)[1].lower() == ".py":
                            last_py = p
                    if last_py:
                        self.model_py_path = last_py
                        self._update_btn_texts()
                        event.acceptProposedAction()
                        return True
                    else:
                        QMessageBox.information(self, "안내", ".py 파일만 드롭할 수 있습니다.")
                        return True
                return False
        return super().eventFilter(obj, event)

    def next_page(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "경고", "데이터셋(.zip)을 선택하세요.")
            return
        if not self.model_py_path:
            QMessageBox.warning(self, "경고", "모델 파일(.py)을 선택하세요.")
            return

        ok, msg = self._wire_uploaded_zip_into_train_model()
        if not ok:
            QMessageBox.critical(self, "실패", msg)
            return

        ok2, msg2 = self._run_train_py_now()
        if ok2:
            QMessageBox.information(self, "학습 완료", msg2)
        else:
            QMessageBox.critical(self, "학습 실패", msg2)
        if self._navigator["go_next"]:
            self._navigator["go_next"]()

    def back_page(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()


    def _wire_uploaded_zip_into_train_model(self):
        try:
            project_root = Path(__file__).resolve().parents[1]          
            train_dir    = project_root / "train_model"
            train_py     = train_dir / "train.py"
            if not train_py.exists():
                raise FileNotFoundError(f"train.py가 없습니다: {train_py}")

            user_zip = Path(self.dataset_path)
            if not user_zip.exists():
                raise FileNotFoundError(f"업로드 zip이 없습니다: {user_zip}")

            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "content").mkdir(parents=True, exist_ok=True) 
            zip1 = train_dir / "pcb_data.zip"
            zip2 = train_dir / "pcb_data2.zip"
            shutil.copy2(user_zip, zip1)
            shutil.copy2(user_zip, zip2)

            run_name = f"yolov8n_{time.strftime('%Y%m%d_%H%M%S')}"

            runtime = project_root / "runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            cfg = {
                "train_py": str(train_py.resolve()),
                "work_dir": str(train_dir.resolve()),  
                "original_zip": str(zip1.resolve()),  
                "augmented_zip": str(zip2.resolve()),
                "data_root": str((train_dir / "content" / "pcb_raw").resolve()),
                "run_name": run_name
            }
            with open(runtime / "run_config.json", "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)

            return True, (
                "데이터셋 연결 완료\n"
                f"- CWD: {cfg['work_dir']}\n"
                f"- pcb_data.zip: {cfg['original_zip']}\n"
                f"- pcb_data2.zip: {cfg['augmented_zip']}\n"
                "※ 학습 시 ./content/pcb_raw 로 압축 해제 → ./content/pcb_std 구성 → runs_ultra/ 하위에 결과 저장"
            )
        except Exception as e:
            return False, f"연결 실패: {e}\n{traceback.format_exc()}"
        
    def next_page(self):
        if not self.dataset_path:
            QMessageBox.warning(self, "경고", "데이터셋(.zip)을 선택하세요.")
            return
        if not self.model_py_path:
            QMessageBox.warning(self, "경고", "모델 파일(.py)을 선택하세요.")
            return

        ok, msg = self._wire_uploaded_zip_into_train_model()
        if not ok:
            QMessageBox.critical(self, "실패", msg)
            return
        QMessageBox.information(self, "완료", msg)


        if self._navigator["go_next"]:
            self._navigator["go_next"]()

    def _run_train_py_now(self):
        project_root = Path(__file__).resolve().parents[1]
        runtime_json = project_root / "runtime" / "run_config.json"
        if not runtime_json.exists():
            return False, f"실행 설정이 없습니다: {runtime_json}"

        with open(runtime_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        py = Path(cfg["train_py"])
        cwd = Path(cfg["work_dir"])
        if not py.exists():
            return False, f"train.py를 찾을 수 없습니다: {py}"
        if not cwd.exists():
            return False, f"CWD가 없습니다: {cwd}"

        log_path = cwd / "train_stdout.log"
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    [sys.executable, str(py)],
                    cwd=str(cwd),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
            code = proc.returncode
            if code == 0:
                return True, f"학습이 정상 종료되었습니다. 로그: {log_path}"
            else:
                return False, f"학습 실패(returncode={code}). 로그 확인: {log_path}"
        except Exception as e:
            return False, f"학습 실행 중 예외: {e}"
        
    def _run_train_in_new_console(self):
        project_root = Path(__file__).resolve().parents[1]
        cfg_path = project_root / "runtime" / "run_config.json"
        if not cfg_path.exists():
            QMessageBox.critical(self, "실패", f"실행 설정이 없습니다: {cfg_path}")
            return

        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        py  = Path(cfg["train_py"])
        cwd = Path(cfg["work_dir"])
        if not py.exists() or not cwd.exists():
            QMessageBox.critical(self, "실패", f"경로 오류\npy={py}\ncwd={cwd}")
            return

        cmd = f'"{sys.executable}" "{py}"'
        subprocess.Popen(["cmd.exe", "/k", cmd], cwd=str(cwd))
        QMessageBox.information(self, "실행됨", f"새 콘솔에서 학습을 시작했습니다.\nCWD: {cwd}\nPY : {py}")

