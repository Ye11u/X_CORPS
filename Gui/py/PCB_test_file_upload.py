import os, sys, shutil, zipfile, json, time, uuid
from pathlib import Path
from PyQt5 import uic
from datetime import datetime
from PyQt5.QtWidgets import QWidget, QFileDialog, QMessageBox
from PyQt5.QtCore import QEvent

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
    
UI_FILE = ROOT / "ui" / "PCB_test_file_upload.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

IM_ROOT          = ROOT / "inference_model"
DATASET_TEST_DIR = IM_ROOT / "content" / "pcb_std" / "images"
WEIGHTS_DIR      = IM_ROOT / "content" / "weights"
WEIGHTS_DST      = WEIGHTS_DIR / "last.pt"
MANIFEST_PATH    = IM_ROOT / "content" / "UI_test_manifest.json"
TEST_SCRIPT      = IM_ROOT / "test.py"
DEFAULT_WEIGHTS_PATH = IM_ROOT / "content" / "weights" / "last.pt"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

def _log(msg: str): print(msg, flush=True)

def _rmtree_retry(p: Path, tries=6, delay=0.25):
    for i in range(tries):
        try:
            if p.exists(): shutil.rmtree(p)
            return
        except PermissionError:
            if i == tries - 1: raise
            time.sleep(delay)

def _atomic_dir_swap(tmp_dir: Path, dst_dir: Path):
    _rmtree_retry(dst_dir)
    os.replace(str(tmp_dir), str(dst_dir))  

CONF_PCT = 25
IOU_PCT  = 70
def _run_dir_name(conf_pct=CONF_PCT, iou_pct=IOU_PCT):
    return f"pred_test_conf{conf_pct}_iou{iou_pct}"

class PCB_Test_FileUpload(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._navigator = {"go_next": None, "go_back": None}
        self.dataset_files = []
        self.t_start = time.perf_counter()
        if DEFAULT_WEIGHTS_PATH.exists():
            self.model_weight_path = str(DEFAULT_WEIGHTS_PATH)
        else:
            self.model_weight_path = None
        self.proc = None
        self.worker = None

        self.file_btn.clicked.connect(self.pick_dataset_zip_or_images)
        self.file_btn_2.clicked.connect(self.pick_model_weight)
        self.next_btn.clicked.connect(self.next_page)
        if hasattr(self, "back_btn"): self.back_btn.clicked.connect(self.back_page)

        self.file_btn.setAcceptDrops(True); self.file_btn_2.setAcceptDrops(True)
        self.file_btn.installEventFilter(self); self.file_btn_2.installEventFilter(self)

        self._btn1_style0 = self.file_btn.styleSheet() or ""
        self._btn2_style0 = self.file_btn_2.styleSheet() or ""
        self._hover_style = "QPushButton{border:2px dashed #1C8CFF;background:#EAF6FF;border-radius:10px;}"
        self._update_btn_texts()
    
    def set_inference_worker(self, worker):
        self.worker = worker
        self.worker.finished.connect(self._on_inference_finished)
        self.worker.log_msg.connect(lambda msg: print(f"[WORKER] {msg}")) 
        self.worker.error_occurred.connect(lambda err: QMessageBox.critical(self, "오류", err))
        
    def set_navigator(self, go_next=None, go_back=None):
        self._navigator["go_next"] = go_next; self._navigator["go_back"] = go_back

    def pick_dataset_zip_or_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Test Data", "",
                    "Zip (*.zip);;Images (*.jpg *.jpeg *.png *.bmp)")
        if files: self.dataset_files = files; self._update_btn_texts()

    def pick_model_weight(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Trained Model", "",
                                              "PyTorch Weights (*.pt *.pth)")
        if path: self.model_weight_path = path; self._update_btn_texts()

    def _update_btn_texts(self):
        self.file_btn.setText(os.path.basename(self.dataset_files[0]) if self.dataset_files and len(self.dataset_files)==1
                              else (f"{len(self.dataset_files)} files selected" if self.dataset_files else
                                    "Select .zip or images (.jpg/.png/.bmp)"))
        self.file_btn_2.setText(os.path.basename(self.model_weight_path) if self.model_weight_path else "Select model (.pt/.pth)")

    def eventFilter(self, obj, event):
        if obj is self.file_btn:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    allow = {'.zip', *IMG_EXTS}
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        if p and Path(p).suffix.lower() in allow:
                            obj.setStyleSheet(self._hover_style); event.acceptProposedAction(); return True
                return False
            if event.type() == QEvent.DragLeave: obj.setStyleSheet(self._btn1_style0); return False
            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn1_style0)
                if event.mimeData().hasUrls():
                    zips, imgs = [], []
                    for u in event.mimeData().urls():
                        p = u.toLocalFile();
                        if not p: continue
                        ext = Path(p).suffix.lower()
                        if ext == ".zip": zips.append(p)
                        elif ext in IMG_EXTS: imgs.append(p)
                    self.dataset_files = [zips[-1]] if zips else sorted(set(imgs))
                    if not self.dataset_files:
                        QMessageBox.information(self, "안내", ".zip 또는 이미지 파일만 드롭하세요."); return True
                    self._update_btn_texts(); event.acceptProposedAction(); return True
                return False
        if obj is self.file_btn_2:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if event.mimeData().hasUrls():
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        if p and Path(p).suffix.lower() in {".pt", ".pth"}:
                            obj.setStyleSheet(self._hover_style); event.acceptProposedAction(); return True
                return False
            if event.type() == QEvent.DragLeave: obj.setStyleSheet(self._btn2_style0); return False
            if event.type() == QEvent.Drop:
                obj.setStyleSheet(self._btn2_style0)
                if event.mimeData().hasUrls():
                    last_w = None
                    for u in event.mimeData().urls():
                        p = u.toLocalFile()
                        if p and Path(p).suffix.lower() in {".pt", ".pth"}: last_w = p
                    if last_w: self.model_weight_path = last_w; self._update_btn_texts(); event.acceptProposedAction(); return True
                    QMessageBox.information(self, "안내", ".pt 또는 .pth만 드롭하세요."); return True
                return False
        return super().eventFilter(obj, event)

    def _stage_test_images(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_dir_name = f"test_{timestamp}"
        self.current_test_dir = DATASET_TEST_DIR / test_dir_name

        tmp_dir = DATASET_TEST_DIR / f"test_tmp_{timestamp}_{uuid.uuid4().hex[:6]}"
        tmp_label_dir = None 

        if tmp_dir.exists(): _rmtree_retry(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        _log(f"[TEST] TMP -> {tmp_dir}")

        if len(self.dataset_files)==1 and Path(self.dataset_files[0]).suffix.lower()==".zip":
            zp = self.dataset_files[0]; _log(f"[UNZIP] {zp}")

            tmp_label_dir = DATASET_TEST_DIR.parent / "labels" / f"test_tmp_{timestamp}_{uuid.uuid4().hex[:6]}"
            if tmp_label_dir.exists(): _rmtree_retry(tmp_label_dir)
            tmp_label_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(zp, 'r') as zf:
                for m in zf.infolist():
                    if m.is_dir(): continue

                    file_path = Path(m.filename)
                    ext = file_path.suffix.lower()

                    if ext in IMG_EXTS:
                        out = tmp_dir / file_path.name
                        with zf.open(m, 'r') as src, open(out, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        _log(f"[EXTRACT IMG] {out.name}")
                    elif ext == ".txt" and "labels" in str(file_path):
                        out = tmp_label_dir / file_path.name
                        with zf.open(m, 'r') as src, open(out, 'wb') as dst:
                            shutil.copyfileobj(src, dst)
                        _log(f"[EXTRACT LBL] {out.name}")
        else:
            _log(f"[COPY] {len(self.dataset_files)} image(s)")
            for p in self.dataset_files:
                p = Path(p)
                if p.exists() and p.suffix.lower() in IMG_EXTS:
                    shutil.copy2(str(p), str(tmp_dir / p.name))

        if not any(p.suffix.lower() in IMG_EXTS for p in tmp_dir.glob("*")):
            _rmtree_retry(tmp_dir)
            if tmp_label_dir: _rmtree_retry(tmp_label_dir)
            raise RuntimeError("테스트 이미지가 없습니다.")

        _log(f"[SWAP] {tmp_dir}  => {self.current_test_dir}")
        _atomic_dir_swap(tmp_dir, self.current_test_dir)

        if tmp_label_dir and any(tmp_label_dir.glob("*.txt")):
            final_label_dir = DATASET_TEST_DIR.parent / "labels" / test_dir_name
            _log(f"[SWAP LBL] {tmp_label_dir} => {final_label_dir}")
            _atomic_dir_swap(tmp_label_dir, final_label_dir)
       
        else:
            print("왜 라벨이 없지!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            final_label_dir = DATASET_TEST_DIR.parent / "labels" / test_dir_name
            _log(f"[SWAP LBL] {tmp_label_dir} => {final_label_dir}")

    def _stage_weights(self):
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_w = WEIGHTS_DIR / f".last_{uuid.uuid4().hex[:6]}.pt"
        _log(f"[WEIGHT] copy -> {tmp_w}")
        shutil.copy2(str(self.model_weight_path), str(tmp_w))
        if WEIGHTS_DST.exists():
            try: _rmtree_retry(WEIGHTS_DST)  
            except Exception: pass
        os.replace(str(tmp_w), str(WEIGHTS_DST))
        _log(f"[WEIGHT] ready -> {WEIGHTS_DST}")

    def _write_manifest(self):
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_images_source": self.dataset_files,
            "weights_source": self.model_weight_path,
            "staged_test_dir": str(self.current_test_dir),
            "staged_weights": str(WEIGHTS_DST),
        }
        MANIFEST_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        _log(f"[MANIFEST] {MANIFEST_PATH}")

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

    def _run_test_script(self):
        if not self.worker:
            QMessageBox.critical(self, "오류", "추론 시스템이 초기화되지 않았습니다.")
            return

        from PyQt5.QtCore import QMetaObject, Q_ARG, Qt
        QMetaObject.invokeMethod(self.worker, "load_model", 
                                 Qt.QueuedConnection, 
                                 Q_ARG(str, self.model_weight_path))
                                 
        self.next_btn.setEnabled(False)
        
        conf_value = getattr(self, 'conf_threshold', 0.25)
        iou_value = getattr(self, 'iou_threshold', 0.70)
        
        self.t_start = time.perf_counter()
        QMetaObject.invokeMethod(self.worker, "run_inference",
                                 Qt.QueuedConnection,
                                 Q_ARG(str, str(self.current_test_dir)),
                                 Q_ARG(float, conf_value),
                                 Q_ARG(float, iou_value))

    def _on_inference_finished(self, output_dir):
        _log(f"[DONE] 결과 경로: {output_dir}")
        self.next_btn.setEnabled(True)

        if self._navigator["go_next"]:
            self._navigator["go_next"](start_time=self.t_start)
            
    def _on_test_finished(self):
        _log("[RUN] test.py 완료")
        self.next_btn.setEnabled(True)

    def next_page(self):
        t_total_start = time.perf_counter()
        if not self.dataset_files:
            QMessageBox.warning(self, "경고", "테스트 데이터(.zip 또는 이미지)를 선택하세요."); return
        if not self.model_weight_path:
            QMessageBox.warning(self, "경고", "훈련된 모델(.pt/.pth)을 선택하세요."); return
        try:
            _log("[NEXT] staging start")
            self._stage_test_images()
            self._stage_weights()
            self._write_manifest()
            _log("[NEXT] staging done")
        except Exception as e:
            QMessageBox.critical(self, "준비 실패", f"준비 중 오류:\n{e}"); return
        if Path(self.model_weight_path).suffix.lower()==".pth":
            QMessageBox.information(self,"안내","가중치가 .pth입니다. YOLO는 보통 .pt를 사용합니다.")
        self._run_test_script()
        
        t_total_end = time.perf_counter()
        print(f"[Next Page] 넘어가는데 걸리는 시간: {t_total_end - t_total_start:.4f}s")

    def back_page(self):
        if self._navigator["go_back"]: self._navigator["go_back"]()