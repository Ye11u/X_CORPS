#######################################################################################
# 추론 결과 화면 >> 이미지 그리드 표시, 필터링, 상세 보기 기능 제공
#######################################################################################
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
UI_FILE = ROOT / "ui" / "PCB_result.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

# 추론 결과에 필요한 설정값 저장
IM_ROOT = ROOT / "inference_model"  # 추론 모델 루트 디렉토리
TEST_SCRIPT = IM_ROOT / "test.py"  # 추론 실행 스크립트 경로
MANIFEST_PATH = IM_ROOT / "content" / "UI_test_manifest.json"  # 테스트 설정 매니페스트 파일 경로

# 추론 결과 디렉토리 경로 (여러 위치에서 검색)
RESULT_ROOTS = [
    IM_ROOT / "infer_results",
    ROOT / "infer_results",    
]
RUNS_ROOT = ROOT / "runs_ultra"  # YOLO 실행 결과 저장 디렉토리

DATASET_TEST_DIR = ROOT / "dataset"  # 테스트 데이터셋 디렉토리

# 클래스 ID와 이름 매핑
IDX2NAME = {0: "silk", 1: "short", 2: "pad_open", 3: "open"}  # 클래스 ID -> 이름 변환 딕셔너리
NAME2IDX = {v: k for k, v in IDX2NAME.items()}  # 클래스 이름 -> ID 변환 딕셔너리
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 지원하는 이미지 확장자

# 그리드 레이아웃 설정
GRID_COLS = 3  # 그리드 열 개수
CELL_MIN_W = 200  # 셀 최소 너비
RIGHT_GUTTER = 8  # 오른쪽 여백

# 추론 실행 시 사용할 기본 임계값 설정
CONF_PCT = 25  # Confidence 임계값 (퍼센트)
IOU_PCT  = 70  # IoU 임계값 (퍼센트)

# 클래스별 배경색 설정
CLASS_COLORS = {
    "normal":   "#A5D6A7",  # 정상 (연한 녹색)
    "silk":     "#D4ACAC",  # 실크 결함 (연한 빨간색)
    "short":    "#CB7575",  # 단락 결함 (빨간색)
    "pad_open": "#CF6679",  # 패드 오픈 결함 (분홍색)
    "open":     "#B75151",  # 오픈 결함 (진한 빨간색)
}

# 텍스트 색상 설정
TEXT_COLORS = {
    "normal": "#2E7D32",  # 정상 텍스트 색상 (진한 녹색)
    "defect": "#C62828",  # 결함 텍스트 색상 (진한 빨간색)
}

def _label_text_color(label_text: str) -> str:
    """라벨 텍스트에 맞는 색상을 반환하는 함수"""
    return TEXT_COLORS["normal"] if label_text == "normal" else TEXT_COLORS["defect"]


def run_dir_name(conf_pct=CONF_PCT, iou_pct=IOU_PCT):
    """추론 결과 디렉토리 이름을 생성하는 함수"""
    return f"pred_test_conf{conf_pct}_iou{iou_pct}"

def find_latest_test_dir():
    """가장 최근의 테스트 디렉토리를 찾는 함수"""
    if not DATASET_TEST_DIR.exists():
        return None

    pattern = re.compile(r'^test_tmp_(\d{14})$')  # 타임스탬프 패턴 매칭
    latest_dir = None
    latest_timestamp = None
    
    for item in DATASET_TEST_DIR.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match:
                timestamp_str = match.group(1)
                try:
                    timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
                    # 가장 최근 타임스탬프를 가진 디렉토리 선택
                    if latest_timestamp is None or timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_dir = item
                except ValueError:
                    continue
    
    return latest_dir

def find_latest_result_dir():
    """가장 최근의 추론 결과 디렉토리와 mAP 값을 찾는 함수"""
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
                    map_value = int(match.group(1))  # mAP 값 추출
                    timestamp_str = match.group(2)  # 타임스탬프 추출
                    try:
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        # 가장 최근 타임스탬프를 가진 디렉토리 선택
                        if latest_timestamp is None or timestamp > latest_timestamp:
                            latest_timestamp = timestamp
                            latest_dir = item
                            latest_map = map_value
                    except ValueError:
                        continue
    return (latest_dir, latest_map) if latest_dir else (None, None)

def _read_manifest_ts():
    """매니페스트 파일에서 타임스탬프를 읽어오는 함수"""
    try:
        d = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        return datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None

def _read_manifest_target():
    """매니페스트 파일에서 대상 이미지 개수를 읽어오는 함수"""
    try:
        d = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        staged = Path(d.get("staged_test_dir", ""))
        if staged.exists():
            # 준비된 테스트 디렉토리의 이미지 파일 개수 반환
            return sum(1 for p in staged.iterdir()
                       if p.is_file() and p.suffix.lower() in IMG_EXTS)
    except Exception:
        pass
    
    # 매니페스트가 없으면 최신 테스트 디렉토리에서 개수 확인
    latest_dir = find_latest_test_dir()
    if latest_dir and latest_dir.exists():
        return sum(1 for p in latest_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in IMG_EXTS)
    
    return 0

def _darken(hex_color: str, factor: float = 0.82) -> str:
    """색을 어둡게 만드는 함수 (테두리 색상 생성용)"""
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
        self.process_start_time = None  # 프로세스 시작 시간 (실행 시간 측정용)
        self._navigator = {"go_back": None}  # 화면 전환을 위한 네비게이터 (초기 화면에서 함수를 주입받음)
        
        # 버튼 클릭 이벤트 연결 
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
        
        # 이미지 로딩을 위한 스레드 풀 설정
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)  # 최대 4개 스레드로 병렬 처리

        self._setup_scroll_area()
        self._setup_file_list_area()

        self.proc = None  # 추론 프로세스 객체
        self.items = []  # 모든 이미지 아이템 리스트
        self.filtered_items = []  # 현재 필터링된 아이템들
        self.thumb_cache = {}  # 썸네일 이미지 캐시
        self.current_filter = "all"  # 현재 적용된 필터
        self.current_search = ""  # 현재 검색어
        self._last_cell_w = None  # 마지막 셀 너비 (폭 변화 없으면 리렌더 생략)

        self._auto_timer = None  # 자동 로드 타이머
        self._autoload_ctx = None  # 자동 로드 컨텍스트 정보 

    # 초기 화면에서 시작 시간을 받아오는 함수 
    def set_start_time(self, t_start):
        self.process_start_time = t_start
        self.calc_total_time()
    
    # 전체 실행 시간을 계산하고 출력하는 함수
    def calc_total_time(self):
        if self.process_start_time:
            t_end = time.perf_counter()
            total_elapsed = t_end - self.process_start_time
            print(f"⏱️ [Total Time] 버튼 클릭 ~ 결과 화면 출력: {total_elapsed:.4f} 초")
            self.process_start_time = None 
            
    # 검색어 변경 시 호출되는 함수
    def _on_search_changed(self, text):
        self.current_search = text.lower().strip()
        self._apply_current_filters()

    # 검색어를 초기화하는 함수
    def _clear_search(self):
        if hasattr(self, "search_edit"):
            self.search_edit.clear()
        self.current_search = ""
        self._apply_current_filters()

    # 현재 필터와 검색어를 적용하여 아이템을 필터링하는 함수
    def _apply_current_filters(self):
        # 클래스 필터 적용
        if self.current_filter == "all":
            filtered_by_class = self.items
        elif self.current_filter == "normal":
            # 정상 이미지 (결함이 없는 경우)
            filtered_by_class = [it for it in self.items if len(it["classes"]) == 0]
        else:
            # 특정 결함 클래스로 필터링
            if self.current_filter not in NAME2IDX:
                filtered_by_class = []
            else:
                cls_id = NAME2IDX[self.current_filter]
                filtered_by_class = [it for it in self.items if cls_id in it["classes"]]
        
        # 검색어 필터 적용
        if self.current_search:
            self.filtered_items = [
                item for item in filtered_by_class 
                if self.current_search in item["path"].name.lower()
            ]
        else:
            self.filtered_items = filtered_by_class

        # 필터링된 결과로 그리드와 파일 리스트 업데이트
        self._rebuild_grid(self.filtered_items)
        self._update_file_list(self.filtered_items) 
        self._log_to_status(f"필터: {self.current_filter}, 검색: '{self.current_search}' -> 표시 {len(self.filtered_items)} / 전체 {len(self.items)}")

    # 초기 화면에서 페이지 이동 함수를 받아오는 함수 
    def set_navigator(self, go_back=None):
        self._navigator["go_back"] = go_back

    # 뒤로 가기 함수 
    def go_back(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    # 화면이 표시될 때 자동 로드를 시작하는 함수
    def showEvent(self, e):
        super().showEvent(e)
        self._start_autoload()

    # 화면이 숨겨질 때 자동 로드를 중지하는 함수
    def hideEvent(self, e):
        self._stop_autoload()
        super().hideEvent(e)

    # 추론 결과를 자동으로 감지하고 로드하는 함수 시작
    def _start_autoload(self):
        latest_test_dir = find_latest_test_dir()
        if latest_test_dir:
            self._log_to_status(f"[AUTO] 최신 테스트 디렉토리 발견: {latest_test_dir}")
        if hasattr(self, "map_label"):
            self.map_label.setText("")

        run_name_prefix = run_dir_name()  # 추론 결과 디렉토리 이름 패턴
        run_ts = _read_manifest_ts()  # 매니페스트에서 타임스탬프 읽기
        target_n = _read_manifest_target()  # 대상 이미지 개수 읽기

        # 결과 디렉토리 찾기
        primary = None
        for base in RESULT_ROOTS:
            for item in base.iterdir():
                if item.is_dir() and item.name.startswith(run_name_prefix):
                    primary = item
                    break
            if primary:
                break
                
        # 결과 디렉토리가 없으면 임시 디렉토리 생성
        if primary is None:
            primary = RESULT_ROOTS[0] / f"{run_name_prefix}_temp"

        # 대체 이미지 디렉토리 경로 설정
        alt_img_dir = RUNS_ROOT / primary.name if primary.name != f"{run_name_prefix}_temp" else RUNS_ROOT / run_name_prefix

        # 자동 로드 컨텍스트 정보 저장
        self._autoload_ctx = {
            "img_dir_primary": primary,  # 주요 이미지 디렉토리
            "labels_dir": primary / "labels",  # 라벨 디렉토리
            "img_dir_alt": alt_img_dir,  # 대체 이미지 디렉토리
            "run_ts": run_ts,  # 실행 타임스탬프
            "target_n": target_n,  # 대상 이미지 개수
            "tries": 0,  # 시도 횟수
            "latest_test_dir": latest_test_dir,  # 최신 테스트 디렉토리
            "run_name_prefix": run_name_prefix,  # 패턴 매칭용
        }

        # 진행 표시줄 설정
        if self._progress:
            if target_n > 0:
                self._progress.setRange(0, target_n)
                self._progress.setValue(0)
            else:
                self._progress.setRange(0, 0)
            self._progress.show()

        self._log_to_status(f"[AUTO] watch start -> primary={primary} | alt={alt_img_dir} | ts={run_ts} | target={target_n}")

        # 타이머 설정 (500ms마다 결과 디렉토리 확인)
        if self._auto_timer is None:
            self._auto_timer = QTimer(self)
            self._auto_timer.setInterval(500)
            self._auto_timer.timeout.connect(self._poll_results)
        if not self._auto_timer.isActive():
            self._auto_timer.start()
        self._poll_results()  # 즉시 한 번 실행


    # 자동 로드를 중지하는 함수
    def _stop_autoload(self):
        if self._auto_timer and self._auto_timer.isActive():
            self._auto_timer.stop()
            self._log_to_status("[AUTO] watch stopped")

    # 결과 디렉토리를 주기적으로 확인하여 이미지를 로드하는 함수
    def _poll_results(self):
        ctx = self._autoload_ctx
        if not ctx:
            return
        ctx["tries"] += 1

        # 최신 결과 디렉토리 찾기
        latest_result_dir, latest_map = find_latest_result_dir()
    
        # 새로운 결과 디렉토리가 발견되면 업데이트
        if latest_result_dir and latest_result_dir != ctx.get("last_checked_dir"):
            self._log_to_status(f"[AUTO] 새로운 결과 디렉토리 발견: {latest_result_dir}")
            ctx["img_dir_primary"] = latest_result_dir
            ctx["img_dir_alt"] = latest_result_dir
            ctx["labels_dir"] = latest_result_dir / "labels"
            ctx["last_checked_dir"] = latest_result_dir

            # mAP 값 표시
            if latest_map is not None and hasattr(self, "map_label"):
                map_value = latest_map * 0.001
                self.map_label.setText(f"{map_value:.3f}")
                self.map_label.setStyleSheet("font-size: 20px; font-weight: bold;")

        primary = ctx["img_dir_primary"]
        # 주요 디렉토리가 없으면 다시 찾기 시도
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

        # 이미지 디렉토리가 없으면 대기
        if not img_dir or not img_dir.exists():
            self._log_to_status(f"[AUTO] waiting... ({ctx['tries']}) 결과 폴더 없음")
            if ctx["tries"] >= 120:  # 60초 후 타임아웃
                QMessageBox.warning(self, "경고", "결과 폴더가 생성되지 않았습니다. test.py 실행 상태를 확인하세요.")
                self._stop_autoload()
            return
        
        # 유효한 이미지 파일 찾기 (타임스탬프 기준 필터링)
        all_imgs = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
        run_ts = ctx["run_ts"]
        if run_ts:
            # 실행 타임스탬프 이후에 생성된 이미지만 유효한 것으로 간주
            valid_imgs = [p for p in all_imgs if datetime.fromtimestamp(p.stat().st_mtime) >= run_ts]
        else:
            valid_imgs = all_imgs
        n = len(valid_imgs)
        target_n = ctx.get("target_n", 0)
        
        # 진행 표시줄 업데이트
        if self._progress:
            if target_n > 0:
                if self._progress.maximum() != target_n:
                    self._progress.setRange(0, target_n)
                self._progress.setValue(min(n, target_n))
            else:
                if self._progress.maximum() != 0:
                    self._progress.setRange(0, 0)

        self._log_to_status(f"[AUTO] scan: {img_dir} | imgs(valid={n}) / target={target_n} | labels={labels_dir}")

        # 대상 이미지 개수에 도달하지 않았으면 계속 대기
        if target_n > 0 and n < target_n:
            return
        if n == 0:
            return
        
        # 모든 이미지가 준비되면 렌더링 시작
        try:
            self._index_and_render_with_list(sorted(valid_imgs), labels_dir=labels_dir, run_ts=run_ts)
            self._apply_current_filters()  
            self._log_to_status(f"[AUTO] render(final): {n} image(s) 로드 완료!")
        finally:
            self._stop_autoload()

    # 추론 스크립트를 실행하는 함수
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
        self.proc.setProcessChannelMode(QProcess.MergedChannels)  # 표준 출력과 에러 출력 병합
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")  # 버퍼링 없이 출력 (실시간 로그 확인)
        self.proc.setProcessEnvironment(env)
        self.proc.readyReadStandardOutput.connect(
            lambda: self._read_stream(self.proc.readAllStandardOutput())
        )
        self.proc.finished.connect(self._on_infer_finished)

        python = sys.executable
        self.proc.setWorkingDirectory(str(TEST_SCRIPT.parent))
        self.proc.start(python, ["-u", str(TEST_SCRIPT)])  # -u: unbuffered 모드

    # 프로세스 출력 스트림을 읽어서 상태 레이블에 표시하는 함수
    def _read_stream(self, data):
        s = bytes(data).decode("utf-8", errors="ignore").rstrip()
        if s:
            print(s, flush=True)
            if hasattr(self, "status_label") and self.status_label:
                old = self.status_label.text()
                self.status_label.setText((old + "\n" if old else "") + s)

    # 추론 프로세스가 종료되었을 때 호출되는 함수
    def _on_infer_finished(self):
        if hasattr(self, "start_infer_btn"):
            self.start_infer_btn.setEnabled(True)
        self._log_to_status("[LOCAL] test.py 종료")
        self._start_autoload()  # 결과 자동 로드 시작

    # 스크롤 영역과 그리드 레이아웃을 설정하는 함수
    def _setup_scroll_area(self):
        from PyQt5.QtWidgets import QWidget
        self._container = QWidget()
        self._grid = QGridLayout(self._container)
        self._grid.setContentsMargins(8, 8, 8, 8)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)

        self.scroll_area_images.setWidget(self._container)
        self.scroll_area_images.setWidgetResizable(True)

        self.scroll_area_images.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 가로 스크롤바 비활성화
        self.scroll_area_images.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 세로 스크롤바 필요시 표시

        self._container.setMinimumWidth(self.scroll_area_images.viewport().width())

    # 윈도우 크기 변경 시 그리드를 재구성하는 함수
    def resizeEvent(self, e):
        super().resizeEvent(e)
        try:
            vsb = self.scroll_area_images.verticalScrollBar()
            sb_w = (vsb.sizeHint().width() if (vsb and vsb.isVisible()) else 0)
            self.scroll_area_images.setViewportMargins(0, 0, sb_w, 0)
            self._container.setMinimumWidth(self.scroll_area_images.viewport().width())
            new_cell_w = self._compute_cell_width()
            # 셀 너비가 변경되었을 때만 그리드 재구성
            if new_cell_w != self._last_cell_w and self.filtered_items:
                self._rebuild_grid(self.filtered_items)
        except Exception:
            pass
    
    # 이미지 경로를 인덱싱하고 그리드와 파일 리스트를 렌더링하는 함수
    def _index_and_render_with_list(self, img_paths, labels_dir: Path = None, run_ts: datetime = None):
        t_start_total = time.perf_counter() 
        self._log_to_status(f"[PERF] Start indexing {len(img_paths)} images...")
        
        self.items.clear()
        self.filtered_items.clear()
        self.thumb_cache.clear()

        # 라벨 파일 파싱
        t_start_labels = time.perf_counter() 
        self._log_to_status("[PERF] Parsing labels...") 

        preds = {}
        if labels_dir and labels_dir.exists():
            txt_files = list(labels_dir.glob("*.txt"))
            for txt in txt_files:
                # 타임스탬프 필터링
                if run_ts:
                    if datetime.fromtimestamp(txt.stat().st_mtime) < run_ts:
                        continue
                try:
                    content = txt.read_text(encoding="utf-8")
                except:
                    continue
                
                # 라벨 파일에서 클래스 ID 추출
                cls_set = set()
                for ln in content.splitlines():
                    parts = ln.split()
                    if parts:
                        try: cls_set.add(int(float(parts[0])))  # 첫 번째 값이 클래스 ID
                        except: pass
                preds[txt.stem] = cls_set

        t_end_labels = time.perf_counter() 
        self._log_to_status(f"[PERF] 1. Label parsing time: {t_end_labels - t_start_labels:.4f} s")
        
        t_start_items = time.perf_counter() 

        sorted_paths = sorted(img_paths)
        
        # 이미지 아이템 생성 (경로, 클래스, 라벨 정보 포함)
        t_start_items = time.perf_counter() 
        self.items = [
            {
                "path": p,
                "classes": preds.get(p.stem, set()),  # 예측된 클래스 ID 집합
                "label": "normal" if not preds.get(p.stem) else ",".join(sorted([IDX2NAME.get(c, str(c)) for c in preds.get(p.stem)]))  # 클래스 이름으로 변환
            }
            for p in sorted_paths
        ]
        self.filtered_items = self.items[:]
        
        t_end_items = time.perf_counter() 
        print(f"[PERF] Index & Render Total: {t_end_items - t_start_items:.4f} s")
        
        # 그리드 재구성
        t_start_grid = time.perf_counter()
        self._rebuild_grid(self.filtered_items)
        t_end_grid = time.perf_counter() 
        self._log_to_status(f"[PERF] 3. Grid rebuild time: {t_end_grid - t_start_grid:.4f} s")
        
        # 파일 리스트 업데이트
        t_start_list = time.perf_counter() 
        self._update_file_list(self.filtered_items)
        t_end_list = time.perf_counter() 
        self._log_to_status(f"[PERF] 4. File list update time: {t_end_list - t_start_list:.4f} s") 

        t_end_total = time.perf_counter() 
        self._log_to_status(f"[PERF] ======= Total UI Render time: {t_end_total - t_start_total:.4f} s =======") 

    # 그리드를 재구성하여 이미지 썸네일을 표시하는 함수
    def _rebuild_grid(self, items):
        # 기존 그리드 아이템 제거
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()

        cell_w = self._compute_cell_width()
        self._last_cell_w = cell_w
        r, c = 0, 0  # 그리드 행, 열 인덱스

        self._container.setUpdatesEnabled(False)  # 업데이트 최적화

        for rec in items:
            p = rec["path"]

            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)

            # 라벨에 따라 배경색 설정
            label_text = rec["label"]
            bg_color = CLASS_COLORS.get("normal") if label_text == "normal" else (CLASS_COLORS.get(label_text, "#F5F5F7") if "," not in label_text else "#F5F5F7")
            border_color = _darken(bg_color)  # 테두리 색상 (배경색보다 어둡게)
            
            frame.setStyleSheet(f"background-color: {bg_color}; border: 3px solid {border_color}; border-radius: 8px;")

            v = QVBoxLayout(frame)
            v.setContentsMargins(6, 6, 6, 6)

            # 클릭 가능한 이미지 라벨 생성
            img_lbl = ClickableImageLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setFixedWidth(cell_w)
            img_lbl.setFixedHeight(int(cell_w * 0.75))  # 4:3 비율
            img_lbl.setStyleSheet("background-color: #E0E0E0; color: #888;")
            img_lbl.setText("Loading...")

            img_lbl.clicked.connect(lambda checked=False, path=p, txt=label_text: self._on_image_click(path, txt))

            # 파일명과 라벨 표시
            cap = QLabel(f"{p.name}\n[{label_text}]")
            cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)
            tcolor = _label_text_color(label_text)
            cap.setStyleSheet(f"color: {tcolor}; background: rgba(255,255,255,0.6); border-radius: 6px; font-size: 12px; font-weight: 600;")

            v.addWidget(img_lbl)
            v.addWidget(cap)
            self._grid.addWidget(frame, r, c)

            # 썸네일 로딩 (캐시 확인 또는 워커로 비동기 로딩)
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
    
    # 썸네일 이미지를 라벨 위젯에 적용하는 함수
    def _apply_pixmap(self, label_widget, pixmap):
        try:
            label_widget.setText("") 
            label_widget.setPixmap(pixmap)
            label_widget.setFixedHeight(pixmap.height()) 
        except RuntimeError:
            pass  # 위젯이 이미 삭제된 경우 무시
    
    # 썸네일 로딩 완료 시 호출되는 함수
    def _on_thumb_loaded(self, path_str, qimage, label_widget, width):
        pixmap = QPixmap.fromImage(qimage)
        key = f"{Path(path_str)}:{width}"
        self.thumb_cache[key] = pixmap  # 캐시에 저장
        self._apply_pixmap(label_widget, pixmap)
    
    # 이미지 클릭 시 상세 보기 다이얼로그를 표시하는 함수
    def _on_image_click(self, path, label_text):
        confidence_data = self._read_confidence_data(
            path, self._autoload_ctx.get("labels_dir") if self._autoload_ctx else None
        )
        dialog = ImagePopupDialog(path, label_text, confidence_data, self)
        dialog.exec_()
        
    # 썸네일 이미지를 가져오는 함수 (캐시 우선)
    def _get_thumb(self, path: Path, target_w: int) -> QPixmap:
        key = f"{path}:{target_w}"
        if key in self.thumb_cache:
            return self.thumb_cache[key]
        
        reader = QImageReader(str(path))
        reader.setScaledSize(QSize(target_w, int(target_w * 0.75)))  # 대략적 비율
        img = reader.read()
        if img.isNull():
            return QPixmap(target_w, int(target_w * 0.75))
            
        pix = QPixmap.fromImage(img)
        self.thumb_cache[key] = pix
        return pix
    
    # 필터를 적용하는 함수
    def apply_filter(self, key: str):
        self.current_filter = key
        self._apply_current_filters()

    # 상태 메시지를 로그와 상태 레이블에 출력하는 함수
    def _log_to_status(self, msg: str):
        print(msg, flush=True)
        if hasattr(self, "status_label") and self.status_label:
            old = self.status_label.text()
            self.status_label.setText((old + "\n" if old else "") + msg)

    # 그리드 셀 너비를 계산하는 함수
    def _compute_cell_width(self) -> int:
        usable = self._viewport_usable_width()
        m_left, m_top, m_right, m_bottom = self._grid.getContentsMargins()
        spacing = self._grid.horizontalSpacing() or 0
        usable -= (m_left + m_right) + spacing * (GRID_COLS - 1)
        w = usable // GRID_COLS if GRID_COLS > 0 else usable
        return max(CELL_MIN_W, int(w))  # 최소 너비 보장

    # 뷰포트 사용 가능한 너비를 계산하는 함수
    def _viewport_usable_width(self) -> int:
        vp = self.scroll_area_images.viewport()
        w = vp.width()
        vsb = self.scroll_area_images.verticalScrollBar()
        sb_w = (vsb.sizeHint().width() if (vsb and vsb.isVisible()) else 0)
        return max(0, w - sb_w - RIGHT_GUTTER)

    
    # 파일 리스트 영역을 설정하는 함수
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
        self.scroll_area_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 가로 스크롤바 비활성화
        self.scroll_area_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 세로 스크롤바 필요시 표시

    # 파일 리스트를 업데이트하는 함수
    def _update_file_list(self, items):
        if not hasattr(self, "_file_list_layout"):
            return

        # 기존 리스트 아이템 제거
        while self._file_list_layout.count():
            item = self._file_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # 파일 개수 표시
        count_label = QLabel(f"{len(items)}개 파일")
        count_label.setStyleSheet("font-weight: bold; padding: 4px; background-color: #F0F0F0;")
        self._file_list_layout.addWidget(count_label)

        # 각 파일에 대한 라벨 생성
        for item in items:
            file_path = item["path"]
            label_text = item["label"]

            # 라벨에 따라 배경색 설정
            if label_text == "normal":
                bg_color = CLASS_COLORS["normal"]
            elif "," in label_text:  # 여러 클래스가 있는 경우
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

    # 라벨 파일에서 confidence 데이터를 읽어오는 함수
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
            # UTF-8 실패 시 CP949로 재시도
            lines = label_file.read_text(encoding="cp949").splitlines()

        # 라벨 파일 파싱 (YOLO 형식: class_id x_center y_center width height confidence)
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 6:  # confidence 정보가 있는 경우
               try:
                   cls_id = int(float(parts[0]))  # 클래스 ID
                   x_center = float(parts[1])  # 바운딩 박스 중심 X
                   y_center = float(parts[2])  # 바운딩 박스 중심 Y
                   width = float(parts[3])  # 바운딩 박스 너비
                   height = float(parts[4])  # 바운딩 박스 높이
                   confidence = float(parts[5])  # 확신도
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