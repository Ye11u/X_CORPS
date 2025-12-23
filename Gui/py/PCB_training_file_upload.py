#######################################################################################
# 훈련 파일 업로드 화면 >> next 버튼 누르면 훈련 스크립트 돌리기 시작
#######################################################################################
import os, shutil, json, traceback, time, sys
import subprocess 
from pathlib import Path
from PyQt5 import uic
import res_rc
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from dragdrop_uploader import FileDropWidget # 파일 드래그 앤 드롭 기능 로드 
from PyQt5.QtCore import QEvent, Qt

# 개발 환경(일반 폴더)과 배포 환경(임시 폴더 _MEIPASS) 모두에서 경로 오류를 방지하기 위한 함수 
if hasattr(sys, '_MEIPASS'):
    ROOT = Path(sys._MEIPASS)
else:
    ROOT = Path(__file__).resolve().parents[1]

# 모듈 import 경로 문제 해결을 위해 시스템 경로에 루트 추가
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

def resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    
# UI 파일 로드 
UI_FILE = ROOT / "ui" / "PCB_training_file_upload.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

class PCB_Training_FileUpload(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._navigator = {"go_next": None, "go_back": None} #화면 전환을 위한 네비게이터 (초기 화면에서 함수를 주입받음)
        self.dataset_path = None # 훈련 데이터셋 경로 (사용자로부터 입력받아서 설정할 변수)
        self.model_py_path = None # 훈련 모델 경로 (사용자로부터 입력받아서 설정할 변수)

        # 버튼 클릭 이벤트 연결 
        self.file_btn.clicked.connect(self.pick_dataset_zip)
        self.file_btn_2.clicked.connect(self.pick_model_py)
        self.next_btn.clicked.connect(self.next_page)
        self.back_btn.clicked.connect(self.back_page)

        self._update_btn_texts() # 파일이 선택되면 버튼 텍스트를 업데이트하는 함수

         # 파일 드래그 앤 드롭 함수 연결 
        self.file_btn.setAcceptDrops(True)
        self.file_btn_2.setAcceptDrops(True) 
        
        # 드래그 시 버튼 스타일 변경 
        self.file_btn.installEventFilter(self)
        self.file_btn_2.installEventFilter(self)

        # 드래그 전/후 버튼 스타일 
        self._btn1_style0 = self.file_btn.styleSheet() or ""
        self._btn2_style0 = self.file_btn_2.styleSheet() or ""
        self._hover_style = """
        QPushButton, QToolButton {
            border: 2px dashed #1C8CFF; 
            background: #EAF6FF;     
            border-radius: 10px;
        }
        """
    
        # 드개그 앤 드랍을 인식하는 영역 생성 
        self.dropper = FileDropWidget(
            text="여기로 .zip(데이터셋) 또는 .py(모델) 드래그",
            accept_exts=['.zip', '.py'],
            accept_dirs=False,  
            recursive=False,
            parent=self
        )
        self.dropper.filesDropped.connect(self._on_files_dropped)

        # UI 구조가 변경되더라도 코드가 깨지지 않도록, 부모 위젯의 레이아웃을 찾아 Dropper를 추가함
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
            # 레이아웃을 못 찾을 경우 추가  (예외 처리, 현 코드 및 ui에선 레이아웃이 이미 정의되어 있음)
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

    # 초기 화면에서 페이지 이동 함수를 받아오는 함수 
    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next
        self._navigator["go_back"] = go_back

    # 기본 타일 선택 함수: 파일 탐색기로 .zip 파일을 선택함 
    def pick_dataset_zip(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Training Dataset (.zip)", "", "Zip Files (*.zip)"
        )
        if path:
            self.dataset_path = path
            self._update_btn_texts()

    # 기본 모델 파일 선택 함수: 파일 탐색기로 .py 함수 선택 
    def pick_model_py(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Model Script (.py)", "", "Python Files (*.py)"
        )
        if path:
            self.model_py_path = path
            self._update_btn_texts()

    # 드랍 영역에 파일이 들어왔을 때 동작하는 함수 
    def _on_files_dropped(self, paths):
        if not paths:
            return

        picked_zip = None
        picked_py = None

        for p in paths:
            # 확장자를 확인하여 zip 형식은 데이터셋으로, py 형식은 모델로 할당 
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

    # 버튼 텍스트를 선택된 파일명으로 변경하는 함수 
    def _update_btn_texts(self):
        self.file_btn.setText(
            os.path.basename(self.dataset_path) if self.dataset_path else "Select .zip dataset"
        )
        self.file_btn_2.setText(
            os.path.basename(self.model_py_path) if self.model_py_path else "Select model .py"
        )
    # 버튼 위에 파일을 드래그했을 때 점선 테두리 디자인을 변경하는 함수  
    def eventFilter(self, obj, event):
        # 데이터셋 버튼 처리
        if obj is self.file_btn:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    for url in event.mimeData().urls():
                        p = url.toLocalFile()
                        # zip 파일인 경우에만 스타일 변경 및 드롭 허용
                        if p and os.path.splitext(p)[1].lower() == ".zip":
                            obj.setStyleSheet(self._hover_style)
                            event.acceptProposedAction()
                            return True
                return False
            # 드래그가 버튼 밖으로 나가거나 드롭이 끝나면 원래 스타일로 복구
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
        # 모델 스크립트 버튼 처리 (.py 형식으로 위와 동일하게 동작)
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

    # 파일 유효성 검사 후 학습 환경 구성
    def next_page(self):
        # 필수 데이터셋 / 모델 파일 선택 여부 확인
        if not self.dataset_path:
            QMessageBox.warning(self, "경고", "데이터셋(.zip)을 선택하세요.")
            return
        if not self.model_py_path:
            QMessageBox.warning(self, "경고", "모델 파일(.py)을 선택하세요.")
            return
        
        # 학습 파일 생성: 선택된 파일들을 학습 폴더로 복사 & json 생성하는 함수 
        ok, msg = self._wire_uploaded_zip_into_train_model() # 생성 성공 여부, 결과 메세지를 반환 받음
        if not ok:
            QMessageBox.critical(self, "실패", msg) # 실패 메세지 반환 시 중단 
            return
        
        # 학습 스크립트 실행 함수 
        ok2, msg2 = self._run_train_py_now()
        if ok2:
            QMessageBox.information(self, "학습 완료", msg2)
        else:
            QMessageBox.critical(self, "학습 실패", msg2)
            
        # 모두 끝나면 다음 페이지로 이동 
        if self._navigator["go_next"]:
            self._navigator["go_next"]()
            
    # 뒤로 가기 함수 
    def back_page(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    # 사용자가 업로드한 파일을 Gui\train_model 디렉토리로 복사해서 사용 
    def _wire_uploaded_zip_into_train_model(self):
        try:
            # Gui\train_model 디렉토리 찾기 
            project_root = Path(__file__).resolve().parents[1]          
            train_dir    = project_root / "train_model"
            train_py     = train_dir / "train.py"
            if not train_py.exists():
                raise FileNotFoundError(f"train.py가 없습니다: {train_py}")

            user_zip = Path(self.dataset_path)
            if not user_zip.exists():
                raise FileNotFoundError(f"업로드 zip이 없습니다: {user_zip}")

            # 원본 파일을 건드리지 않고, 사본을 생성하여 사용 
            train_dir.mkdir(parents=True, exist_ok=True)
            (train_dir / "content").mkdir(parents=True, exist_ok=True) 
            zip1 = train_dir / "pcb_data.zip"
            zip2 = train_dir / "pcb_data2.zip"
            shutil.copy2(user_zip, zip1)
            shutil.copy2(user_zip, zip2)

            # 고유 폴더명 생성 (timestamp 이용) >> 결과가 겹치는 것을 방지 
            run_name = f"yolov8n_{time.strftime('%Y%m%d_%H%M%S')}"

            runtime = project_root / "runtime"
            runtime.mkdir(parents=True, exist_ok=True)
            cfg = {
                "train_py": str(train_py.resolve()), # 절대 경로로 변환하여 저장 
                "work_dir": str(train_dir.resolve()),  
                "original_zip": str(zip1.resolve()),  
                "augmented_zip": str(zip2.resolve()),
                "data_root": str((train_dir / "content" / "pcb_raw").resolve()),
                "run_name": run_name
            }
            # 훈련 설정 JSON 파일을 생성 
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
    
    # 학습을 수행하는 함수 >> subprocess에서 학습을 돌리고 로그 캡쳐 
    def _run_train_py_now(self):
        project_root = Path(__file__).resolve().parents[1]
        runtime_json = project_root / "runtime" / "run_config.json"
        if not runtime_json.exists():
            return False, f"실행 설정이 없습니다: {runtime_json}"

        # 생성한 json 설정 파일 불러옴 
        with open(runtime_json, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        py = Path(cfg["train_py"])
        cwd = Path(cfg["work_dir"])
        if not py.exists():
            return False, f"train.py를 찾을 수 없습니다: {py}"
        if not cwd.exists():
            return False, f"CWD가 없습니다: {cwd}"

        # 콘솔에 출력되는 로그를 읽어옴 
        log_path = cwd / "train_stdout.log"
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run( # subprocess: 외부 파이썬 스크립트 실행 
                    [sys.executable, str(py)], # sys.executable: subprocess에서 현재 GUI를 실행 중인 환경과 동일한 환경 사용
                    cwd=str(cwd), # 작업 디렉토리 설정 
                    stdout=lf, # 표준 출력을 파일로 저장 
                    stderr=subprocess.STDOUT, # 에러메세지도 같이 저장 
                    text=True,
                    check=False
                )
            # 프로세스 종료 코드 변수 
            code = proc.returncode 
            if code == 0: # 0이면 정상 종료
                return True, f"학습이 정상 종료되었습니다. 로그: {log_path}"
            else: # 아니면 오류 
                return False, f"학습 실패(returncode={code}). 로그 확인: {log_path}"
        except Exception as e:
            return False, f"학습 실행 중 예외: {e}"
    