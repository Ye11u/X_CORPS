import sys, json, re, csv, yaml, os
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt5.QtWidgets import QWidget, QMessageBox, QVBoxLayout, QApplication
import pyqtgraph as pg

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
    
UI_FILE = ROOT / "ui" / "PCB_training_monitoring.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

TARGET_KEYS = [
    "train/box_loss", "train/cls_loss", "train/dfl_loss",
    "metrics/precision(B)", "metrics/recall(B)",
    "val/box_loss", "val/cls_loss", "val/dfl_loss",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
]

_ansi = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
def _clean_ansi(s: str) -> str:
    return _ansi.sub('', s)

_epoch_re = re.compile(r'(?i)\bepoch[^0-9]*(\d+)\s*/\s*(\d+)\b')

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')
pg.setConfigOptions(antialias=True) 


class PCB_Training_Monitoring(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self._navigator = {"go_test": None, "go_back": None}

        self.grid = pg.GraphicsLayoutWidget()
        lay = self.widget.layout()
        if lay is None:
            lay = QVBoxLayout(self.widget)
            self.widget.setLayout(lay)
        lay.addWidget(self.grid)

        self.plots, self.curves = [], []
        titles = [
            "train/box_loss", "train/cls_loss", "train/dfl_loss",
            "metrics/precision(B)", "metrics/recall(B)",
            "val/box_loss", "val/cls_loss", "val/dfl_loss",
            "metrics/mAP50(B)", "metrics/mAP50-95(B)",
        ]

        blue_thick_pen = pg.mkPen('#1f77b4', width=3) 

        for i, title in enumerate(titles):
            if i % 5 == 0 and i != 0:
                self.grid.nextRow()
            p = self.grid.addPlot(title=title)
            p.showGrid(x=True, y=True, alpha=0.2)
            p.setLabel('bottom', 'Epoch')
            c = p.plot([], [], pen=blue_thick_pen)    
            self.plots.append(p)
            self.curves.append(c)
        self.proc = None

        self.cfg = None
        self.work_dir = None
        self.train_py = None

        self.run_name = "yolov8n"
        self.runs_dir = None
        self.results_csv = None

        self._epochs = []  
        self._data = {k: [] for k in TARGET_KEYS} 

        if hasattr(self, "test_btn"):
            self.test_btn.clicked.connect(self.go_test_upload)
        if hasattr(self, "back_btn"):
            self.back_btn.clicked.connect(self.go_back)

        self._locked_run_dir = False
        self._last_csv_mtime = 0

        self.csv_timer = QTimer(self)
        self.csv_timer.setInterval(700)   
        self.csv_timer.timeout.connect(self._pull_results_csv)

    def set_navigator(self, go_test=None, go_back=None):
        self._navigator["go_test"] = go_test
        self._navigator["go_back"] = go_back

    def go_test_upload(self):
        if self._navigator["go_test"]:
            self._navigator["go_test"]()

    def go_back(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    def showEvent(self, e):
        super().showEvent(e)
        if self.proc is None:
            self._start_training()

    def _start_training(self):
        cfg_path = ROOT / "runtime" / "run_config.json"
        if not cfg_path.exists():
            QMessageBox.critical(self, "실패", f"실행 설정이 없습니다: {cfg_path}")
            return

        self.cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        self.work_dir = Path(self.cfg["work_dir"])
        self.train_py = Path(self.cfg["train_py"])
        self.run_name = self.cfg.get("run_name", "yolov8n")
        self.runs_dir = self.work_dir / "runs_ultra" / self.run_name
        self.results_csv = self.runs_dir / "results.csv"

        if not self.work_dir.exists() or not self.train_py.exists():
            QMessageBox.critical(self, "실패",
                                f"경로 오류\nwork_dir={self.work_dir}\ntrain_py={self.train_py}")
            return

        if not hasattr(self, "logText") or self.logText is None:
            try:
                from PyQt5.QtWidgets import QTextEdit
                self.logText = QTextEdit(self)
                self.logText.setReadOnly(True)
                container = self.widget.parentWidget() or self
                (container.layout() or self.layout()).addWidget(self.logText)
            except Exception:
                pass

        self._pb_total = None
        self._pb_last_closed = 0  
        self._pb_pending_epoch = None 

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(self.work_dir))
        self.proc.setProcessChannelMode(QProcess.MergedChannels)

        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1") 
        env.insert("NO_COLOR", "1")        
        self.proc.setProcessEnvironment(env)

        self.proc.readyReadStandardOutput.connect(self._on_stdout)
        self.proc.finished.connect(self._on_finished)

        self.proc.start(sys.executable, ["-u", str(self.train_py)])
        if not self.proc.waitForStarted(3000):
            QMessageBox.critical(self, "실패", "train.py 실행 시작 실패")
            self.proc = None
            return


    def _on_stdout(self):
        raw = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")

        if hasattr(self, "logText") and self.logText is not None:
            self.logText.moveCursor(self.logText.textCursor().End)
            self.logText.insertPlainText(raw)
            self.logText.moveCursor(self.logText.textCursor().End)

        for line in raw.splitlines():
            s = _clean_ansi(line.strip())
            if not s:
                continue

            if s.startswith("@@RUN_DIR "):
                try:
                    p = json.loads(s[9:])
                    run_dir = Path(p.get("run_dir", "")).resolve()
                    self._lock_and_set_run_dir(run_dir) 
                    te = self._read_total_epochs_from_args(self.runs_dir)
                    if te and getattr(self, "progressBar", None):
                        self._pb_total = te
                        self.progressBar.setRange(0, te)
                    if getattr(self, "logText", None):
                        self.logText.append(f"[monitor] run_dir set&locked: {self.runs_dir}")
                except Exception:
                    pass
                continue


            if "Logging results to" in s:
                if self._locked_run_dir:   
                    continue
                try:
                    part = s.split("Logging results to", 1)[1].strip()
                    cand = Path(part).resolve()
                    self.runs_dir = cand
                    self.results_csv = cand / "results.csv"
                    te = self._read_total_epochs_from_args(self.runs_dir)
                    if te and getattr(self, "progressBar", None):
                        self._pb_total = te
                        self.progressBar.setRange(0, te)
                    if getattr(self, "logText", None):
                        self.logText.append(f"[monitor] run_dir from stdout: {self.runs_dir}")
                    if not self.csv_timer.isActive():
                        self.csv_timer.start()
                except Exception:
                    pass
                continue

            if s.startswith("@@EPOCH "):
                try:
                    p = json.loads(s[8:])
                    cur, tot = int(p.get("cur", 0)), int(p.get("tot", 0))
                    if getattr(self, "progressBar", None):
                        if not getattr(self, "_pb_total", None) or self._pb_total < tot:
                            self._pb_total = tot
                            self.progressBar.setRange(0, tot)
                        safe_val = cur if cur < self._pb_total else self._pb_total - 1
                        self.progressBar.setValue(max(0, min(safe_val, self._pb_total - 1)))
                except Exception:
                    pass
                continue

            if s.startswith("@@EPOCH_LOSS "):
                try:
                    p = json.loads(s[13:])
                    ep = int(p.get("epoch", 0))
                    if ep <= 0:
                        continue
                    self._ensure_epoch_slot(ep)
                    for k in ("train/box_loss", "train/cls_loss", "train/dfl_loss"):
                        if k in p and p[k] is not None:
                            self._data[k][ep - 1] = float(p[k])
                    self._redraw_all_curves()
                except Exception:
                    pass
                continue

            if s.startswith("@@ACC "):
                try:
                    p = json.loads(s[6:])
                    ep = int(p.get("epoch", 0))
                    if ep <= 0:
                        continue
                    self._ensure_epoch_slot(ep)
                    mapping = {
                        "metrics/precision(B)": "precision",
                        "metrics/recall(B)"   : "recall",
                        "metrics/mAP50(B)"    : "mAP50",
                        "metrics/mAP50-95(B)" : "mAP50-95",
                        "val/box_loss"        : "val/box_loss",
                        "val/cls_loss"        : "val/cls_loss",
                        "val/dfl_loss"        : "val/dfl_loss",
                    }
                    for tgt, src in mapping.items():
                        if src in p and p[src] is not None:
                            self._data[tgt][ep - 1] = float(p[src])


                    if getattr(self, "progressBar", None) and getattr(self, "_pb_total", None):
                        self.progressBar.setValue(min(self.progressBar.value(), self._pb_total - 1))

                    self._redraw_all_curves()
                    self._update_from_results_csv(ep)
                except Exception:
                    pass
                continue

    def _on_finished(self, *args):
        try:
            leftover = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="ignore")
            if leftover and hasattr(self, "logText") and self.logText is not None:
                self.logText.append(leftover)
        except Exception:
            pass

        if getattr(self, "progressBar", None):
            if self._pb_total:
                self.progressBar.setRange(0, self._pb_total)
                self.progressBar.setValue(self._pb_total)
            else:
                self.progressBar.setValue(self.progressBar.maximum())
        if getattr(self, "proc", None):
            try:
                self.proc.deleteLater()
            except Exception:
                pass
            self.proc = None

        QMessageBox.information(self, "완료", "학습이 종료되었습니다.")


    def _ensure_epoch_slot(self, ep: int):
        while len(self._epochs) < ep:
            self._epochs.append(len(self._epochs) + 1)
            for k in self._data:
                self._data[k].append(None)

    def _redraw_all_curves(self):
        for i, k in enumerate(TARGET_KEYS):
            ys_all = self._data[k]
            xs = [e for e, y in zip(self._epochs, ys_all) if y is not None]
            ys = [y for y in ys_all if y is not None]
            self.curves[i].setData(xs, ys)
            QApplication.processEvents()


    def _update_from_results_csv(self, ep: int):
        try:
            if not self.results_csv or not self.results_csv.exists():
                return
            target_row = None
            with open(self.results_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    try:
                        e = int(float(r.get("epoch", 0)))
                    except Exception:
                        continue
                    if e == ep:
                        target_row = r
            if not target_row:
                return

            self._ensure_epoch_slot(ep)
            mapping = {
                "metrics/precision(B)": "metrics/precision(B)",
                "metrics/recall(B)"   : "metrics/recall(B)",
                "metrics/mAP50(B)"    : "metrics/mAP50(B)",
                "metrics/mAP50-95(B)" : "metrics/mAP50-95(B)",
                "val/box_loss"        : "val/box_loss",
                "val/cls_loss"        : "val/cls_loss",
                "val/dfl_loss"        : "val/dfl_loss",
            }
            for tgt, src in mapping.items():
                v = target_row.get(src, "")
                if v not in ("", None):
                    try:
                        self._data[tgt][ep - 1] = float(v)
                    except Exception:
                        pass

            self._redraw_all_curves()
        except Exception:
            pass

    def _read_total_epochs_from_args(self, run_dir: Path) -> int:
        for name in ("args.yaml", "train_args.yaml"):
            p = Path(run_dir) / name
            if p.exists():
                try:
                    data = yaml.safe_load(p.read_text(encoding="utf-8"))
                    ep = int(data.get("epochs", 0) or 0)
                    if ep > 0:
                        return ep
                except Exception:
                    pass
        return 0

    def _lock_and_set_run_dir(self, run_dir: Path):
        self.runs_dir = Path(run_dir)
        self.results_csv = self.runs_dir / "results.csv"
        self._locked_run_dir = True
        if not self.csv_timer.isActive():
            self.csv_timer.start()


    def _pull_results_csv(self):
        try:
            if not self.results_csv or not self.results_csv.exists():
                return
            mtime = self.results_csv.stat().st_mtime
            if mtime == self._last_csv_mtime:
                return
            self._last_csv_mtime = mtime

            with open(self.results_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                return

            alias = {
                "train/box_loss":       ["train/box_loss","box_loss"],
                "train/cls_loss":       ["train/cls_loss","cls_loss"],
                "train/dfl_loss":       ["train/dfl_loss","dfl_loss"],
                "val/box_loss":         ["val/box_loss"],
                "val/cls_loss":         ["val/cls_loss"],
                "val/dfl_loss":         ["val/dfl_loss"],
                "metrics/precision(B)": ["metrics/precision(B)","metrics/precision","precision"],
                "metrics/recall(B)":    ["metrics/recall(B)","metrics/recall","recall"],
                "metrics/mAP50(B)":     ["metrics/mAP50(B)","metrics/mAP50","mAP50","map50"],
                "metrics/mAP50-95(B)":  ["metrics/mAP50-95(B)","metrics/mAP50-95","mAP50-95","map"],
            }

            for r in rows:
                try:
                    ep = int(float(r.get("epoch", 0)))
                except Exception:
                    continue
                if ep <= 0:
                    continue
                self._ensure_epoch_slot(ep)
                for tgt, names in alias.items():
                    for name in names:
                        v = r.get(name, "")
                        if v not in ("", None):
                            try:
                                self._data[tgt][ep-1] = float(v)
                                break
                            except Exception:
                                pass

            self._redraw_all_curves()
        except Exception:
            pass

