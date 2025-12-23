#######################################################################################
# 테스트 파일 업로드 >> next btn 누르면 추론 스크립트 돌리는 화면 
#######################################################################################
import os, sys, shutil, zipfile, json, time, uuid
from pathlib import Path
from PyQt5 import uic
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QEvent

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
UI_FILE = ROOT / "ui" / "PCB_test_file_upload.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

# 추론에 필요한 설정값 저장
IM_ROOT          = ROOT / "inference_model"  # 추론 모델 루트 디렉토리
DATASET_TEST_DIR = IM_ROOT / "content" / "pcb_std" / "images"  # 테스트 이미지 저장 디렉토리
WEIGHTS_DIR      = IM_ROOT / "content" / "weights"  # 모델 가중치 저장 디렉토리
WEIGHTS_DST      = WEIGHTS_DIR / "last.pt"  # 최종 사용할 가중치 파일 경로
MANIFEST_PATH    = IM_ROOT / "content" / "UI_test_manifest.json"  # 테스트 설정 매니페스트 파일 경로
TEST_SCRIPT      = IM_ROOT / "test.py"  # 추론 실행 스크립트 경로
DEFAULT_WEIGHTS_PATH = IM_ROOT / "content" / "weights" / "last.pt"  # 기본 가중치 파일 경로
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 지원하는 이미지 확장자

def _log(msg: str): 
    """로그 메시지를 출력하는 함수"""
    print(msg, flush=True)

def _rmtree_retry(p: Path, tries=6, delay=0.25):
    """디렉토리 삭제를 재시도하는 함수"""
    for i in range(tries):
        try:
            if p.exists(): shutil.rmtree(p)
            return
        except PermissionError: # 권한 오류시 재시도
            if i == tries - 1: raise
            time.sleep(delay)

def _atomic_dir_swap(tmp_dir: Path, dst_dir: Path):
    """임시 디렉토리를 최종 디렉토리로 교체하는 함수"""
    _rmtree_retry(dst_dir)
    os.replace(str(tmp_dir), str(dst_dir))  

# 추론 실행 시 사용할 기본 임계값 설정
CONF_PCT = 25  # Confidence 임계값 (퍼센트)
IOU_PCT  = 70  # IoU 임계값 (퍼센트)

def _run_dir_name(conf_pct=CONF_PCT, iou_pct=IOU_PCT):
    """추론 결과 디렉토리 이름을 생성하는 함수"""
    return f"pred_test_conf{conf_pct}_iou{iou_pct}"

class PCB_Test_FileUpload(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._navigator = {"go_next": None, "go_back": None}  # 화면 전환을 위한 네비게이터 (초기 화면에서 함수를 주입받음)
        self.dataset_files = []  # 테스트 데이터셋 파일 경로 리스트 (사용자로부터 입력받아서 설정할 변수)
        self.t_start = time.perf_counter()  # 추론 시작 시간 측정용
        # 기본 가중치 파일이 존재하면 자동으로 설정
        if DEFAULT_WEIGHTS_PATH.exists():
            self.model_weight_path = str(DEFAULT_WEIGHTS_PATH)
        else:
            self.model_weight_path = None  # 모델 가중치 경로 (사용자로부터 입력받아서 설정할 변수)
        self.proc = None  # 서브프로세스 객체 
        self.worker = None  # 추론 작업을 수행하는 워커 객체

        # 버튼 클릭 이벤트 연결 
        self.file_btn.clicked.connect(self.pick_dataset_zip_or_images)
        self.file_btn_2.clicked.connect(self.pick_model_weight)
        self.next_btn.clicked.connect(self.next_page)
        if hasattr(self, "back_btn"): self.back_btn.clicked.connect(self.back_page)

        # 파일 드래그 앤 드롭 기능 활성화
        self.file_btn.setAcceptDrops(True); self.file_btn_2.setAcceptDrops(True)
        self.file_btn.installEventFilter(self); self.file_btn_2.installEventFilter(self)

        # 드래그 전/후 버튼 스타일 저장
        self._btn1_style0 = self.file_btn.styleSheet() or ""
        self._btn2_style0 = self.file_btn_2.styleSheet() or ""
        self._hover_style = "QPushButton{border:2px dashed #1C8CFF;background:#EAF6FF;border-radius:10px;}"  # 드래그 시 버튼 스타일
        self._update_btn_texts()  # 파일이 선택되면 버튼 텍스트를 업데이트하는 함수
    
    # 초기 화면에서 추론 워커를 받아오는 함수 
    def set_inference_worker(self, worker):
        self.worker = worker
        self.worker.finished.connect(self._on_inference_finished)  # 추론 완료 시 호출될 함수 연결
        self.worker.log_msg.connect(lambda msg: print(f"[WORKER] {msg}"))  # 워커 로그 메시지 출력
        self.worker.error_occurred.connect(lambda err: QMessageBox.critical(self, "오류", err))  # 오류 발생 시 메시지 박스 표시
        
    # 초기 화면에서 페이지 이동 함수를 받아오는 함수 
    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next; self._navigator["go_back"] = go_back

    # 기본 파일 선택 함수: 파일 탐색기로 .zip 파일 또는 이미지 파일들을 선택함 
    def pick_dataset_zip_or_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Data", "",
                    "Zip (*.zip);;Images (*.jpg *.jpeg *.png *.bmp)")
        if files: self.dataset_files = files; self._update_btn_texts()

    # 기본 모델 가중치 선택 함수: 파일 탐색기로 .pt 또는 .pth 파일 선택 
    def pick_model_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "",
                                              "PyTorch Weights (*.pt *.pth)")
        if path: self.model_weight_path = path; self._update_btn_texts()

    # 버튼 텍스트를 선택된 파일명으로 변경하는 함수 
    def _update_btn_texts(self):
        self.file_btn.setText(os.path.basename(self.dataset_files[0]) if self.dataset_files and len(self.dataset_files)==1
                              else (f"{len(self.dataset_files)} files selected" if self.dataset_files else
                                    "Select .zip or images (.jpg/.png/.bmp)"))
        self.file_btn_2.setText(os.path.basename(self.model_weight_path) if self.model_weight_path else "Select model (.pt/.pth)")

    # 버튼 위에 파일을 드래그했을 때 점선 테두리 디자인을 변경하는 함수  
    def eventFilter(self, obj, event):
        # 데이터셋 버튼 처리
        if obj is self.file_btn:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    allow = {'.zip', *IMG_EXTS}  # 허용할 확장자 목록
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        # zip 파일 또는 이미지 파일인 경우에만 스타일 변경 및 드롭 허용
                        if p and Path(p).suffix.lower() in allow:
                            obj.setStyleSheet(self._hover_style)
                            event.acceptProposedAction()
                            return True
                return False
            # 드래그가 버튼 밖으로 나가면 원래 스타일로 복구
            if event.type() == QEvent.DragLeave: obj.setStyleSheet(self._btn1_style0); return False
            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn1_style0)  # 드롭 후 원래 스타일로 복구
                if event.mimeData().hasUrls():
                    zips, imgs = [], []
                    for u in event.mimeData().urls():
                        p = u.toLocalFile();
                        if not p: continue
                        ext = Path(p).suffix.lower()
                        if ext == ".zip": zips.append(p)
                        elif ext in IMG_EXTS: imgs.append(p)
                    # zip 파일이 있으면 zip만 사용, 없으면 이미지 파일들 사용
                    self.dataset_files = [zips[-1]] if zips else sorted(set(imgs))
                    if not self.dataset_files:
                        QMessageBox.information(self, "안내", ".zip 또는 이미지 파일만 드롭하세요."); return True
                    self._update_btn_texts(); event.acceptProposedAction(); return True
                return False
        # 모델 가중치 버튼 처리 (.pt/.pth 형식으로 위와 동일하게 동작)
        if obj is self.file_btn_2:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        # .pt 또는 .pth 파일인 경우에만 스타일 변경 및 드롭 허용
                        if p and Path(p).suffix.lower() in {".pt", ".pth"}:
                            obj.setStyleSheet(self._hover_style); event.acceptProposedAction(); return True
                return False
            # 드래그가 버튼 밖으로 나가면 원래 스타일로 복구
            if event.type() == QEvent.DragLeave: obj.setStyleSheet(self._btn2_style0); return False
            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn2_style0)  # 원래 스타일로 복구
                if event.mimeData().hasUrls():
                    last_w = None
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        if p and Path(p).suffix.lower() in {".pt", ".pth"}: last_w = p
                    if last_w: self.model_weight_path = last_w; self._update_btn_texts(); event.acceptProposedAction(); return True
                    QMessageBox.information(self, "안내", ".pt 또는 .pth만 드롭하세요."); return True
                return False
        return super().eventFilter(obj, event)

    # 테스트 이미지를 추론 디렉토리로 준비하는 함수 (zip 압축 해제 또는 이미지 복사)
    def _stage_test_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir_name = f"test_{timestamp}"
        self.current_test_dir = DATASET_TEST_DIR / test_dir_name  # 최종 테스트 디렉토리 경로

        # 임시 디렉토리 생성 (고유한 이름으로 충돌 방지)
        tmp_dir = DATASET_TEST_DIR / f"test_tmp_{timestamp}_{uuid.uuid4().hex[:6]}"
        tmp_label_dir = None  # 라벨 파일용 임시 디렉토리

        if tmp_dir.exists(): _rmtree_retry(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[TEST] TMP -> {tmp_dir}")

        # zip 파일인 경우 압축 해제
        if len(self.dataset_files)==1 and Path(self.dataset_files[0]).suffix.lower()==".zip":
            zp = self.dataset_files[0]; _log(f"[UNZIP] {zp}")

            # 라벨 파일용 임시 디렉토리 생성
            tmp_label_dir = DATASET_TEST_DIR.parent / "labels" / f"test_tmp_{timestamp}_{uuid.uuid4().hex[:6]}"
            if tmp_label_dir.exists(): _rmtree_retry(tmp_label_dir)
            tmp_label_dir.mkdir(parents=True, exist_ok=True)

            # zip 파일에서 이미지와 라벨 파일 추출
            with zipfile.ZipFile(zp, 'r') as zf:
                for m in zf.infolist():
                    if m.is_dir(): continue

                    file_path = Path(m.filename)
                    ext = file_path.suffix.lower()

                    # 이미지 파일 추출
                    if ext in IMG_EXTS:
                        out = tmp_dir / file_path.name
                        with zf.open(m, 'r') as src, open(out, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        _log(f"[EXTRACT IMG] {out.name}")
                    # 라벨 파일 추출 (경로에 "labels"가 포함된 경우)
                    elif ext == ".txt" and "labels" in str(file_path):
                        out = tmp_label_dir / file_path.name
                        with zf.open(m, 'r') as src, open(out, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        _log(f"[EXTRACT LBL] {out.name}")
        else:
            # 이미지 파일들을 직접 복사
            _log(f"[COPY] {len(self.dataset_files)} image(s)")
            for p in self.dataset_files:
                p = Path(p)
                if p.exists() and p.suffix.lower() in IMG_EXTS:
                    shutil.copy2(str(p), str(tmp_dir / p.name))

        # 이미지 파일이 없으면 오류 발생
        if not any(p.suffix.lower() in IMG_EXTS for p in tmp_dir.glob("*")):
            _rmtree_retry(tmp_dir)
            if tmp_label_dir: _rmtree_retry(tmp_label_dir)
            raise RuntimeError("테스트 이미지가 없습니다.")

        # 임시 디렉토리를 최종 디렉토리로 교체
        _log(f"[SWAP] {tmp_dir}  => {self.current_test_dir}")
        _atomic_dir_swap(tmp_dir, self.current_test_dir)

        # 라벨 디렉토리도 교체
        if tmp_label_dir and any(tmp_label_dir.glob("*.txt")):
            final_label_dir = DATASET_TEST_DIR.parent / "labels" / test_dir_name
            _log(f"[SWAP LBL] {tmp_label_dir} => {final_label_dir}")
            _atomic_dir_swap(tmp_label_dir, final_label_dir)
       
        else:
            # 라벨이 없는 경우에도 디렉토리 생성 (디버그용 메시지)
            print("라벨 없음")
            final_label_dir = DATASET_TEST_DIR.parent / "labels" / test_dir_name
            _log(f"[SWAP LBL] {tmp_label_dir} => {final_label_dir}")

    # 모델 가중치 파일을 추론 디렉토리로 준비하는 함수
    def _stage_weights(self):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        # 임시 파일명으로 복사 (고유한 이름으로 충돌 방지)
        tmp_w = WEIGHTS_DIR / f".last_{uuid.uuid4().hex[:6]}.pt"
        _log(f"[WEIGHT] copy -> {tmp_w}")
        shutil.copy2(str(self.model_weight_path), str(tmp_w))
        # 기존 가중치 파일이 있으면 삭제 시도
        if WEIGHTS_DST.exists():
            try: _rmtree_retry(WEIGHTS_DST)  
            except Exception: pass
        # 임시 파일을 최종 파일로 교체
        os.replace(str(tmp_w), str(WEIGHTS_DST))
        _log(f"[WEIGHT] ready -> {WEIGHTS_DST}")

    # 테스트 설정을 매니페스트 파일로 저장하는 함수
    def _write_manifest(self):
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),  # 타임스탬프
            "test_images_source": self.dataset_files,  # 원본 테스트 이미지 경로
            "weights_source": self.model_weight_path,  # 원본 가중치 파일 경로
            "staged_test_dir": str(self.current_test_dir),  # 준비된 테스트 디렉토리 경로
            "staged_weights": str(WEIGHTS_DST),  # 준비된 가중치 파일 경로
        }
        MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"[MANIFEST] {MANIFEST_PATH}")

    # 이전 추론 결과를 정리하는 함수 (정적 메서드)
    def _cleanup_previous_results():
        run_name = _run_dir_name()
        roots = [
            IM_ROOT / "infer_results",
            ROOT / "infer_results",
            ROOT / "runs_ultra",
        ]
        for base in roots:
            target = base / run_name
            if target.exists():
                _log(f"[CLEAN] remove previous: {target}")
                _rmtree_retry(target)

    # 추론 스크립트를 실행하는 함수 (워커를 통해 비동기로 실행)
    def _run_test_script(self):
        if not self.worker:
            QMessageBox.critical(self, "오류", "추론 시스템이 초기화되지 않았습니다.")
            return

        from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
        # 워커에 모델 로드 요청 (비동기)
        QMetaObject.invokeMethod(self.worker, "load_model", 
                                 Qt.QueuedConnection, 
                                 Q_ARG(str, self.model_weight_path))
                                 
        self.next_btn.setEnabled(False)  # 추론 중에는 버튼 비활성화
        
        # Confidence 및 IoU 임계값 가져오기 (기본값 사용)
        conf_value = getattr(self, 'conf_threshold', 0.25)
        iou_value = getattr(self, 'iou_threshold', 0.70)
        
        self.t_start = time.perf_counter()  # 추론 시작 시간 기록
        # 워커에 추론 실행 요청 (비동기)
        QMetaObject.invokeMethod(self.worker, "run_inference",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, str(self.current_test_dir)),
                                 Q_ARG(float, conf_value),
                                 Q_ARG(float, iou_value))

    # 추론 완료 시 호출되는 콜백 함수
    def _on_inference_finished(self, output_dir):
        _log(f"[DONE] 결과 경로: {output_dir}")
        self.next_btn.setEnabled(True)  # 버튼 다시 활성화

        # 다음 페이지로 이동 (시작 시간 전달)
        if self._navigator["go_next"]:
            self._navigator["go_next"](start_time=self.t_start)
            
    # 테스트 스크립트 완료 시 호출되는 함수 (실행시간 측정용)
    def _on_test_finished(self):
        _log("[RUN] test.py 완료")
        self.next_btn.setEnabled(True)

    # 파일 유효성 검사 후 추론 환경 구성 및 실행
    def next_page(self):
        t_total_start = time.perf_counter()
        # 필수 테스트 데이터 / 모델 가중치 파일 선택 여부 확인
        if not self.dataset_files:
            QMessageBox.warning(self, "경고", "테스트 데이터(.zip 또는 이미지)를 선택하세요."); return
        if not self.model_weight_path:
            QMessageBox.warning(self, "경고", "훈련된 모델(.pt/.pth)을 선택하세요."); return
        try:
            _log("[NEXT] staging start")
            self._stage_test_images()  # 테스트 이미지 준비
            self._stage_weights()  # 가중치 파일 준비
            self._write_manifest()  # 매니페스트 파일 작성
            _log("[NEXT] staging done")
        except Exception as e:
            QMessageBox.critical(self, "준비 실패", f"준비 중 오류:\n{e}"); return
        # .pth 파일 사용 시 경고 메시지 표시
        if Path(self.model_weight_path).suffix.lower()==".pth":
            QMessageBox.information(self,"안내","가중치가 .pth입니다. YOLO는 보통 .pt를 사용합니다.")
        self._run_test_script()  # 추론 스크립트 실행
        
        t_total_end = time.perf_counter()
        print(f"[Next Page] 넘어가는데 걸리는 시간: {t_total_end - t_total_start:.4f}s")

    # 뒤로 가기 함수 
    def back_page(self):
        if self._navigator["go_back"]: self._navigator["go_back"]()