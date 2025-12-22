import sys, json, re, os
from datetime import datetime
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QLabel, QMessageBox, QGridLayout, QVBoxLayout, QFrame, QLineEdit, QHBoxLayout, QPushButton
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit
from PyQt5.QtCore import pyqtSignal
import time
from PyQt5.QtGui import QImage, QImageReader
from PyQt5.QtCore import QThreadPool, QRunnable, QObject, pyqtSlot, QSize


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

UI_FILE = ROOT / "ui" / "PCB_result.ui"

FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

IM_ROOT = ROOT / "inference_model"
TEST_SCRIPT = IM_ROOT / "test.py"
MANIFEST_PATH = IM_ROOT / "content" / "UI_test_manifest.json"

RESULT_ROOTS = [
    IM_ROOT / "infer_results",
    ROOT / "infer_results",    
]
RUNS_ROOT = ROOT / "runs_ultra" 

DATASET_TEST_DIR = ROOT / "dataset" 

IDX2NAME = {0: "silk", 1: "short", 2: "pad_open", 3: "open"}
NAME2IDX = {v: k for k, v in IDX2NAME.items()}
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

GRID_COLS = 3     
CELL_MIN_W = 200    
RIGHT_GUTTER = 8    

CONF_PCT = 25
IOU_PCT  = 70

CLASS_COLORS = {
    "normal":   "#A5D6A7",  
    "silk":     "#D4ACAC",  
    "short":    "#CB7575",  
    "pad_open": "#CF6679",
    "open":     "#B75151",  
}

TEXT_COLORS = {
    "normal": "#2E7D32",  
    "defect": "#C62828", 
}

def _label_text_color(label_text: str) -> str:
    return TEXT_COLORS["normal"] if label_text == "normal" else TEXT_COLORS["defect"]


def run_dir_name(conf_pct=CONF_PCT, iou_pct=IOU_PCT):
    return f"pred_test_conf{conf_pct}_iou{iou_pct}"

def find_latest_test_dir():
    if not DATASET_TEST_DIR.exists():
        return None

    pattern = re.compile(r'^test_tmp_(\d{14})$')  
    latest_dir = None
    latest_timestamp = None
    
    for item in DATASET_TEST_DIR.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_dir = item
                except ValueError:
                    continue
    
    return latest_dir

def find_latest_result_dir():
    pattern = re.compile(r'^pred_test_conf\d+_iou\d+_mAP(\d{4})_(\d{8}_\d{6})$')
    latest_dir = None
    latest_timestamp = None
    latest_map = None
    
    for base in RESULT_ROOTS:
        if not base.exists():
            continue
            
        for item in base.iterdir():
            if item.is_dir():
                match = pattern.match(item.name)
                if match:
                    map_value = int(match.group(1))
                    timestamp_str = match.group(2)
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        if latest_timestamp is None or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            latest_dir = item
                            latest_map = map_value
                    except ValueError:
                        continue
    return (latest_dir, latest_map) if latest_dir else (None, None)

def _read_manifest_ts():
    try:
        d = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _read_manifest_target():
    try:
        d = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        staged = Path(d.get("staged_test_dir", ""))
        if staged.exists():
            return sum(1 for p in staged.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
    except Exception:
        pass
    
    latest_dir = find_latest_test_dir()
    if latest_dir and latest_dir.exists():
        return sum(1 for p in latest_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    
    return 0

def _darken(hex_color: str, factor: float = 0.82) -> str:
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"



class PCB_Result(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.process_start_time = None
        self._navigator = {"go_back": None}
        
        if hasattr(self, "back_btn"):
            self.back_btn.clicked.connect(self.go_back)
        if hasattr(self, "start_infer_btn"):
            self.start_infer_btn.clicked.connect(self.start_infer)
        if hasattr(self, "all_btn"):
            self.all_btn.clicked.connect(lambda: self.apply_filter("all"))
        if hasattr(self, "normal_btn"):
            self.normal_btn.clicked.connect(lambda: self.apply_filter("normal"))
        if hasattr(self, "open_btn"):
            self.open_btn.clicked.connect(lambda: self.apply_filter("open"))
        if hasattr(self, "pad_open_btn"):
            self.pad_open_btn.clicked.connect(lambda: self.apply_filter("pad_open"))
        if hasattr(self, "short_btn"):
            self.short_btn.clicked.connect(lambda: self.apply_filter("short"))
        if hasattr(self, "silk_btn"):
            self.silk_btn.clicked.connect(lambda: self.apply_filter("silk"))
        if not hasattr(self, "scroll_area_images"):
            raise RuntimeError("UI에 QScrollArea 'scroll_area_images'가 필요합니다.")
        self._setup_scroll_area()
        if hasattr(self, "search_edit"):
            self.search_edit.textChanged.connect(self._on_search_changed)
        self._progress = getattr(self, "progressBar", None)
        if self._progress:
            self._progress.hide()
        
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        self._setup_scroll_area()
        self._setup_file_list_area()

        self.proc = None
        self.items = []
        self.filtered_items = []  # 현재 필터링된 아이템들
        self.thumb_cache = {}
        self.current_filter = "all"
        self.current_search = ""  # 현재 검색어
        self._last_cell_w = None  # 폭 변화 없으면 리렌더 생략

        self._auto_timer = None
        self._autoload_ctx = None 

    def set_start_time(self, t_start):
        self.process_start_time = t_start
        self.calc_total_time()
    
    def calc_total_time(self):
        if self.process_start_time:
            t_end = time.perf_counter()
            total_elapsed = t_end - self.process_start_time
            print(f"⏱️ [Total Time] 버튼 클릭 ~ 결과 화면 출력: {total_elapsed:.4f} 초")
            self.process_start_time = None 
            
    def _on_search_changed(self, text):
        self.current_search = text.lower().strip()
        self._apply_current_filters()

    def _clear_search(self):
        if hasattr(self, "search_edit"):
            self.search_edit.clear()
        self.current_search = ""
        self._apply_current_filters()

    def _apply_current_filters(self):
        if self.current_filter == "all":
            filtered_by_class = self.items
        elif self.current_filter == "normal":
            filtered_by_class = [it for it in self.items if len(it["classes"]) == 0]
        else:
            if self.current_filter not in NAME2IDX:
                filtered_by_class = []
            else:
                cls_id = NAME2IDX[self.current_filter]
                filtered_by_class = [it for it in self.items if cls_id in it["classes"]]
        
        if self.current_search:
            self.filtered_items = [
                item for item in filtered_by_class 
                if self.current_search in item["path"].name.lower()
            ]
        else:
            self.filtered_items = filtered_by_class

        self._rebuild_grid(self.filtered_items)
        self._update_file_list(self.filtered_items) 
        self._log_to_status(f"필터: {self.current_filter}, 검색: '{self.current_search}' -> 표시 {len(self.filtered_items)} / 전체 {len(self.items)}")

    def set_navigator(self, go_back=None):
        self._navigator["go_back"] = go_back

    def go_back(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    def showEvent(self, e):
        super().showEvent(e)
        self._start_autoload()

    def hideEvent(self, e):
        self._stop_autoload()
        super().hideEvent(e)

    def _start_autoload(self):
        latest_test_dir = find_latest_test_dir()
        if latest_test_dir:
            self._log_to_status(f"[AUTO] 최신 테스트 디렉토리 발견: {latest_test_dir}")
        if hasattr(self, "map_label"):
            self.map_label.setText("")

        run_name_prefix = run_dir_name() 
        run_ts = _read_manifest_ts()
        target_n = _read_manifest_target()

        primary = None
        for base in RESULT_ROOTS:
            for item in base.iterdir():
                if item.is_dir() and item.name.startswith(run_name_prefix):
                    primary = item
                    break
            if primary:
                break
                
        if primary is None:
            primary = RESULT_ROOTS[0] / f"{run_name_prefix}_temp"

        alt_img_dir = RUNS_ROOT / primary.name if primary.name != f"{run_name_prefix}_temp" else RUNS_ROOT / run_name_prefix

        self._autoload_ctx = {
            "img_dir_primary": primary,
            "labels_dir": primary / "labels",
            "img_dir_alt": alt_img_dir,
            "run_ts": run_ts,
            "target_n": target_n,
            "tries": 0,
            "latest_test_dir": latest_test_dir,
            "run_name_prefix": run_name_prefix,  # 패턴 매칭용
        }

        if self._progress:
            if target_n > 0:
                self._progress.setRange(0, target_n)
                self._progress.setValue(0)
            else:
                self._progress.setRange(0, 0)
            self._progress.show()

        self._log_to_status(f"[AUTO] watch start -> primary={primary} | alt={alt_img_dir} | ts={run_ts} | target={target_n}")

        if self._auto_timer is None:
            self._auto_timer = QTimer(self)
            self._auto_timer.setInterval(500)
            self._auto_timer.timeout.connect(self._poll_results)
        if not self._auto_timer.isActive():
            self._auto_timer.start()
        self._poll_results()


    def _stop_autoload(self):
        if self._auto_timer and self._auto_timer.isActive():
            self._auto_timer.stop()
            self._log_to_status("[AUTO] watch stopped")

    def _poll_results(self):
        ctx = self._autoload_ctx
        if not ctx:
            return
        ctx["tries"] += 1

        latest_result_dir, latest_map = find_latest_result_dir()
    
        if latest_result_dir and latest_result_dir != ctx.get("last_checked_dir"):
            self._log_to_status(f"[AUTO] 새로운 결과 디렉토리 발견: {latest_result_dir}")
            ctx["img_dir_primary"] = latest_result_dir
            ctx["img_dir_alt"] = latest_result_dir
            ctx["labels_dir"] = latest_result_dir / "labels"
            ctx["last_checked_dir"] = latest_result_dir

            if latest_map is not None and hasattr(self, "map_label"):
                map_value = latest_map * 0.001
                self.map_label.setText(f"{map_value:.3f}")
                self.map_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        primary = ctx["img_dir_primary"]
        if not primary.exists():
            run_name_prefix = ctx.get("run_name_prefix", "pred_test_conf25_iou70")
            found_dir = None
            for base in RESULT_ROOTS:
                if base.exists():
                    for item in base.iterdir():
                        if item.is_dir() and item.name.startswith(run_name_prefix):
                            found_dir = item
                            break
                if found_dir:
                    break
                    
            if found_dir:
                img_dir = found_dir
                labels_dir = found_dir / "labels"
            else:
                img_dir = None
                labels_dir = None
        else:
            img_dir = primary
            labels_dir = ctx["labels_dir"] if ctx["labels_dir"].exists() else (primary / "labels")

        if not img_dir or not img_dir.exists():
            self._log_to_status(f"[AUTO] waiting... ({ctx['tries']}) 결과 폴더 없음")
            if ctx["tries"] >= 120:
                QMessageBox.warning(self, "경고", "결과 폴더가 생성되지 않았습니다. test.py 실행 상태를 확인하세요.")
                self._stop_autoload()
            return
        
        all_imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        run_ts = ctx["run_ts"]
        if run_ts:
            valid_imgs = [p for p in all_imgs if datetime.fromtimestamp(p.stat().st_mtime) >= run_ts]
        else:
            valid_imgs = all_imgs
        n = len(valid_imgs)
        target_n = ctx.get("target_n", 0)
        if self._progress:
            if target_n > 0:
                if self._progress.maximum() != target_n:
                    self._progress.setRange(0, target_n)
                self._progress.setValue(min(n, target_n))
            else:
                if self._progress.maximum() != 0:
                    self._progress.setRange(0, 0)

        self._log_to_status(f"[AUTO] scan: {img_dir} | imgs(valid={n}) / target={target_n} | labels={labels_dir}")

        if target_n > 0 and n < target_n:
            return
        if n == 0:
            return
        try:
            self._index_and_render_with_list(sorted(valid_imgs), labels_dir=labels_dir, run_ts=run_ts)
            self._apply_current_filters()  
            self._log_to_status(f"[AUTO] render(final): {n} image(s) 로드 완료!")
        finally:
            self._stop_autoload()

    def start_infer(self):
        if not TEST_SCRIPT.exists():
            QMessageBox.critical(self, "오류", f"test.py를 찾을 수 없습니다:\n{TEST_SCRIPT}")
            return
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "안내", "이미 추론이 진행 중입니다.")
            return
        if hasattr(self, "start_infer_btn"):
            self.start_infer_btn.setEnabled(False)

        self._log_to_status("추론 시작 (inference_model/test.py 실행).")
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(
            lambda: self._read_stream(self.proc.readAllStandardOutput())
        )
        self.proc.finished.connect(self._on_infer_finished)

        python = sys.executable
        self.proc.setWorkingDirectory(str(TEST_SCRIPT.parent))
        self.proc.start(python, ["-u", str(TEST_SCRIPT)])

    def _read_stream(self, data):
        s = bytes(data).decode("utf-8", errors="ignore").rstrip()
        if s:
            print(s, flush=True)
            if hasattr(self, "status_label") and self.status_label:
                old = self.status_label.text()
                self.status_label.setText((old + "\n" if old else "") + s)

    def _on_infer_finished(self):
        if hasattr(self, "start_infer_btn"):
            self.start_infer_btn.setEnabled(True)
        self._log_to_status("[LOCAL] test.py 종료")
        self._start_autoload()

    def _setup_scroll_area(self):
        from PyQt5.QtWidgets import QWidget
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)

        self.scroll_area_images.setWidget(self._container)
        self.scroll_area_images.setWidgetResizable(True)

        self.scroll_area_images.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_images.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self._container.setMinimumWidth(self.scroll_area_images.viewport().width())

    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            vsb = self.scroll_area_images.verticalScrollBar()
            sb_w = (vsb.sizeHint().width() if (vsb and vsb.isVisible()) else 0)
            self.scroll_area_images.setViewportMargins(0, 0, sb_w, 0)
            self._container.setMinimumWidth(self.scroll_area_images.viewport().width())
            new_cell_w = self._compute_cell_width()
            if new_cell_w != self._last_cell_w and self.filtered_items:
                self._rebuild_grid(self.filtered_items)
        except Exception:
            pass
    def _index_and_render_with_list(self, img_paths, labels_dir: Path = None, run_ts: datetime = None):
        t_start_total = time.perf_counter() 
        self._log_to_status(f"[PERF] Start indexing {len(img_paths)} images...")
        
        self.items.clear()
        self.filtered_items.clear()
        self.thumb_cache.clear()

        t_start_labels = time.perf_counter() 
        self._log_to_status("[PERF] Parsing labels...") 

        preds = {}
        if labels_dir and labels_dir.exists():
            txt_files = list(labels_dir.glob("*.txt"))
            for txt in txt_files:
                if run_ts:
                    if datetime.fromtimestamp(txt.stat().st_mtime) < run_ts:
                        continue
                try:
                    content = txt.read_text(encoding="utf-8")
                except:
                    continue
                
                cls_set = set()
                for ln in content.splitlines():
                    parts = ln.split()
                    if parts:
                        try: cls_set.add(int(float(parts[0])))
                        except: pass
                preds[txt.stem] = cls_set

        t_end_labels = time.perf_counter() 
        self._log_to_status(f"[PERF] 1. Label parsing time: {t_end_labels - t_start_labels:.4f} s") # 
        
        t_start_items = time.perf_counter() 

        sorted_paths = sorted(img_paths)
        
        t_start_items = time.perf_counter() 
        self.items = [
            {
                "path": p,
                "classes": preds.get(p.stem, set()),
                "label": "normal" if not preds.get(p.stem) else ",".join(sorted([IDX2NAME.get(c, str(c)) for c in preds.get(p.stem)]))
            }
            for p in sorted_paths
        ]
        self.filtered_items = self.items[:]
        
        t_end_items = time.perf_counter() 
        print(f"[PERF] Index & Render Total: {t_end_items - t_start_items:.4f} s")
        
        t_start_grid = time.perf_counter()
        self._rebuild_grid(self.filtered_items)
        t_end_grid = time.perf_counter() 
        self._log_to_status(f"[PERF] 3. Grid rebuild time: {t_end_grid - t_start_grid:.4f} s")
        
        t_start_list = time.perf_counter() 
        self._update_file_list(self.filtered_items)
        t_end_list = time.perf_counter() 
        self._log_to_status(f"[PERF] 4. File list update time: {t_end_list - t_start_list:.4f} s") 

        t_end_total = time.perf_counter() 
        self._log_to_status(f"[PERF] ======= Total UI Render time: {t_end_total - t_start_total:.4f} s =======") 

    def _rebuild_grid(self, items):
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

        cell_w = self._compute_cell_width()
        self._last_cell_w = cell_w
        r, c = 0, 0

        self._container.setUpdatesEnabled(False)

        for rec in items:
            p = rec["path"]

            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)

            label_text = rec["label"]
            bg_color = CLASS_COLORS.get("normal") if label_text == "normal" else (CLASS_COLORS.get(label_text, "#F5F5F7") if "," not in label_text else "#F5F5F7")
            border_color = _darken(bg_color)
            
            frame.setStyleSheet(f"background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 8px;")

            v = QVBoxLayout(frame)
            v.setContentsMargins(6, 6, 6, 6)

            img_lbl = ClickableImageLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFixedWidth(cell_w)
            img_lbl.setFixedHeight(int(cell_w * 0.75)) 
            img_lbl.setStyleSheet("background-color: #E0E0E0; color: #888;")
            img_lbl.setText("Loading...")

            img_lbl.clicked.connect(lambda checked=False, path=p, txt=label_text: self._on_image_click(path, txt))

            cap = QLabel(f"{p.name}\n[{label_text}]")
            cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)
            tcolor = _label_text_color(label_text)
            cap.setStyleSheet(f"color: {tcolor}; background: rgba(255,255,255,0.6); border-radius: 6px; font-size: 12px; font-weight: 600;")

            v.addWidget(img_lbl)
            v.addWidget(cap)
            self._grid.addWidget(frame, r, c)

            key = f"{p}:{cell_w}"
            if key in self.thumb_cache:
                self._apply_pixmap(img_lbl, self.thumb_cache[key])
            else:
                worker = ImageLoadWorker(p, cell_w)
                worker.signals.finished.connect(lambda path_str, img, label_widget=img_lbl: self._on_thumb_loaded(path_str, img, label_widget, cell_w))
                self.thread_pool.start(worker)

            c += 1
            if c >= GRID_COLS:
                c = 0
                r += 1

        self._container.adjustSize()
        self._container.setUpdatesEnabled(True) 
        self.calc_total_time()
    
    def _apply_pixmap(self, label_widget, pixmap):
        try:
            label_widget.setText("") 
            label_widget.setPixmap(pixmap)
            label_widget.setFixedHeight(pixmap.height()) 
        except RuntimeError:
            pass 
    
    def _on_thumb_loaded(self, path_str, qimage, label_widget, width):
        pixmap = QPixmap.fromImage(qimage)
        key = f"{Path(path_str)}:{width}"
        self.thumb_cache[key] = pixmap
        self._apply_pixmap(label_widget, pixmap)
    
    def _on_image_click(self, path, label_text):
        confidence_data = self._read_confidence_data(
            path, self._autoload_ctx.get("labels_dir") if self._autoload_ctx else None
        )
        dialog = ImagePopupDialog(path, label_text, confidence_data, self)
        dialog.exec_()
        
    def _get_thumb(self, path: Path, target_w: int) -> QPixmap:
        key = f"{path}:{target_w}"
        if key in self.thumb_cache:
            return self.thumb_cache[key]
        
        reader = QImageReader(str(path))
        reader.setScaledSize(QSize(target_w, int(target_w * 0.75))) # 대략적 비율
        img = reader.read()
        if img.isNull():
            return QPixmap(target_w, int(target_w * 0.75))
            
        pix = QPixmap.fromImage(img)
        self.thumb_cache[key] = pix
        return pix
    
    def apply_filter(self, key: str):
        self.current_filter = key
        self._apply_current_filters()

    def _log_to_status(self, msg: str):
        print(msg, flush=True)
        if hasattr(self, "status_label") and self.status_label:
            old = self.status_label.text()
            self.status_label.setText((old + "\n" if old else "") + msg)

    def _compute_cell_width(self) -> int:
        usable = self._viewport_usable_width()
        m_left, m_top, m_right, m_bottom = self._grid.getContentsMargins()
        spacing = self._grid.horizontalSpacing() or 0
        usable -= (m_left + m_right) + spacing * (GRID_COLS - 1)
        w = usable // GRID_COLS if GRID_COLS > 0 else usable
        return max(CELL_MIN_W, int(w))

    def _viewport_usable_width(self) -> int:
        vp = self.scroll_area_images.viewport()
        w = vp.width()
        vsb = self.scroll_area_images.verticalScrollBar()
        sb_w = (vsb.sizeHint().width() if (vsb and vsb.isVisible()) else 0)
        return max(0, w - sb_w - RIGHT_GUTTER)

    
    def _setup_file_list_area(self):
        if not hasattr(self, "scroll_area_list"):
            print("Warning: scroll_area_list가 UI에 없습니다.")
            return
   
        self._file_list_container = QWidget()
        self._file_list_layout = QVBoxLayout(self._file_list_container)
        self._file_list_layout.setContentsMargins(8, 8, 8, 8)
        self._file_list_layout.setSpacing(2)

        self.scroll_area_list.setWidget(self._file_list_container)
        self.scroll_area_list.setWidgetResizable(True)
        self.scroll_area_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def _update_file_list(self, items):
        if not hasattr(self, "_file_list_layout"):
            return

        while self._file_list_layout.count():
            item = self._file_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        count_label = QLabel(f"{len(items)}개 파일")
        count_label.setStyleSheet("font-weight: bold; padding: 4px; background-color: #F0F0F0;")
        self._file_list_layout.addWidget(count_label)

        for item in items:
            file_path = item["path"]
            label_text = item["label"]

            if label_text == "normal":
                bg_color = CLASS_COLORS["normal"]
            elif "," in label_text:  
                bg_color = "#F8F8F8"  
            else:
                bg_color = CLASS_COLORS.get(label_text, "#FFFFFF")

            file_label = QLabel(f"{file_path.name}")
            file_label.setStyleSheet(f"""
                QLabel {{
                    color: black;               
                    background-color: {bg_color}; 
                    border: 1px solid #CCCCCC;
                    border-radius: 3px;
                    padding: 4px 8px;
                    margin: 1px;
                    font-size: 11px;
                }}
                QLabel:hover {{
                    border: 2px solid #0078D4;
                }}
            """)

            file_label.setWordWrap(True)
            file_label.setToolTip(f"파일: {file_path.name}\n클래스: {label_text}")
            self._file_list_layout.addWidget(file_label)
        from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
        spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        self._file_list_layout.addItem(spacer)

    def _read_confidence_data(self, image_path, labels_dir):
        if not labels_dir or not labels_dir.exists():
            return None
    
        label_file = labels_dir / f"{image_path.stem}.txt"
        if not label_file.exists():
            return None

        confidence_data = []
        try:
            lines = label_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            lines = label_file.read_text(encoding="cp949").splitlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:  
               try:
                   cls_id = int(float(parts[0]))
                   x_center = float(parts[1])
                   y_center = float(parts[2])
                   width = float(parts[3])
                   height = float(parts[4])
                   confidence = float(parts[5])
                   bbox = f"({x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f})"
                   confidence_data.append((cls_id, confidence, bbox))
               except (ValueError, IndexError):
                   continue
                   
        return confidence_data if confidence_data else None



class ImagePopupDialog(QDialog):
    def __init__(self, image_path, label_text, confidence_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"상세 보기 - {image_path.name}")
        self.setModal(True)
        self.resize(1000, 800)

        layout = QVBoxLayout(self)
        title = QLabel(f"{image_path.name}  [{label_text}]")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(f"""
            QLabel {{
                color: {_label_text_color(label_text)};  
                font-size: 30px;                    
                font-weight: 700;
                padding: 6px 8px;
            }}
        """)
        layout.addWidget(title)
        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(str(image_path))
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(scaled_pixmap)

        confidence_label = QLabel()
        confidence_label.setAlignment(Qt.AlignCenter)   
        confidence_label.setWordWrap(True)
        confidence_label.setStyleSheet(f"""
            QLabel {{
                background-color: #F5F5F5;
                border: 1px solid #CCCCCC;
                padding: 12px;
                font-family: 'Noto Sans KR', '맑은 고딕', sans-serif;
                font-size: 20px;
                font-weight: bold;
                color: {_label_text_color(label_text)};
            }}
        """)

        if confidence_data:
            text_content = f"파일명: {image_path.name}\n\n검출된 객체:\n"
            for i, (cls_id, conf, bbox) in enumerate(confidence_data, 1):
                cls_name = IDX2NAME.get(cls_id, f"Unknown({cls_id})")
                text_content += f"{i}. 클래스: {cls_name} | 확신도: {conf:.3f} | 위치: {bbox}\n"
        else:
            text_content = f"파일명: {image_path.name}\n\n검출된 객체: 없음 (Normal)"

        confidence_label.setText(text_content)

        layout.addWidget(image_label)
        layout.addWidget(confidence_label)

class ImageLoadWorkerSignals(QObject):
    finished = pyqtSignal(str, QImage) 

class ImageLoadWorker(QRunnable):
    def __init__(self, path, target_w):
        super().__init__()
        self.path = path
        self.target_w = target_w
        self.signals = ImageLoadWorkerSignals()

    @pyqtSlot()
    def run(self):
        reader = QImageReader(str(self.path))

        orig_size = reader.size()
        if orig_size.isValid() and orig_size.width() > 0:
            aspect_ratio = orig_size.height() / orig_size.width()
            target_h = int(self.target_w * aspect_ratio)

            reader.setScaledSize(QSize(self.target_w, target_h))
            
            img = reader.read()
            if not img.isNull():
                self.signals.finished.emit(str(self.path), img)
                

class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)