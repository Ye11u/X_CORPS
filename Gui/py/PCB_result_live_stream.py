#######################################################################################
# 라이브 스트리밍 결과 화면 >> 이미지 병합(8x4), 필터링, 상세 보기(자 포함) 기능 제공
#######################################################################################
import sys, json, re, os
import cv2
import numpy as np
from pathlib import Path
from PyQt5 import uic
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QTimer
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtWidgets import (QWidget, QLabel, QMessageBox, QGridLayout, QVBoxLayout, 
                             QFrame, QDialog, QSizePolicy, QSpacerItem)

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
UI_FILE = ROOT / "ui" / "PCB_live_result.ui"
FormClass, BaseClass = uic.loadUiType(str(UI_FILE))

# 라이브 스트리밍 결과에 필요한 설정값 저장
IM_ROOT = ROOT / "inference_model"  # 추론 모델 루트 디렉토리
TEST_SCRIPT = IM_ROOT / "test.py"  # 추론 실행 스크립트 경로
MANIFEST_PATH = IM_ROOT / "content" / "UI_test_manifest.json"  # 테스트 설정 매니페스트 파일 경로

LIVE_TEST_DIR = IM_ROOT / "content" / "pcb_std" / "images" / "test_live_streaming"  # 라이브 스트리밍 테스트 디렉토리
SIMULATION_CACHE_DIR = IM_ROOT / "infer_results" / "simulation_cache"  # 시뮬레이션 캐시 디렉토리

SIMULATION_INTERVAL = 5000  # 시뮬레이션 간격 (밀리초)
DATASET_TEST_DIR = ROOT / "dataset"  # 테스트 데이터셋 디렉토리

# 클래스 ID와 이름 매핑
IDX2NAME = {0: "silk", 1: "short", 2: "pad_open", 3: "open"}  # 클래스 ID -> 이름 변환 딕셔너리
NAME2IDX = {v: k for k, v in IDX2NAME.items()}  # 클래스 이름 -> ID 변환 딕셔너리
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # 지원하는 이미지 확장자

# 그리드 레이아웃 설정
GRID_COLS = 1  # 그리드 열 개수 (라이브 스트리밍은 1열로 표시)
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
    "abnormal": "#CF6679",  # 비정상 (분홍색)
}

# 텍스트 색상 설정
TEXT_COLORS = {
    "normal": "#2E7D32",  # 정상 텍스트 색상 (진한 녹색)
    "defect": "#C62828",  # 결함 텍스트 색상 (진한 빨간색)
}

def _label_text_color(label_text: str) -> str:
    """라벨 텍스트에 맞는 색상을 반환하는 함수"""
    return TEXT_COLORS["normal"] if label_text == "Normal" else TEXT_COLORS["defect"]

def _darken(hex_color: str, factor: float = 0.82) -> str:
    """16진수 색상을 어둡게 만드는 함수 (테두리 색상 생성용)"""
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    r = max(0, min(255, int(r * factor)))
    g = max(0, min(255, int(g * factor)))
    b = max(0, min(255, int(b * factor)))
    return f"#{r:02X}{g:02X}{b:02X}"

def _stitch_images_8x4(img_paths: list, save_path: Path):
    """32개의 이미지를 8x4 그리드로 병합하는 함수"""
    if len(img_paths) == 0:
        _log_to_status("[STITCH] 병합할 이미지가 없습니다.")
        return False
    # 파일명의 숫자를 기준으로 정렬 (자연 정렬)
    key_func = lambda p: [int(c) if c.isdigit() else c for c in re.compile(r'(\d+)').split(p.name)]
    sorted_paths = sorted(img_paths, key=key_func)

    # 32개가 안 되면 None으로 채움
    if len(sorted_paths) < 32:
        sorted_paths.extend([None] * (32 - len(sorted_paths)))

    # 첫 번째 유효한 이미지로 기본 크기 결정
    first_img = None
    for p in sorted_paths:
        if p and p.exists():
            first_img = cv2.imread(str(p))
            if first_img is not None:
                break
                
    if first_img is None:
        _log_to_status(f"[STITCH] 유효한 이미지를 찾을 수 없어 크기를 결정할 수 없습니다.")
        return False
        
    base_h, base_w, base_c = first_img.shape
    placeholder = np.zeros((base_h, base_w, base_c), dtype=np.uint8) + 50  # 회색 플레이스홀더
    
    all_rows = []
    
    try:
        # 4행 8열 그리드로 이미지 배치
        for i in range(4):  # 4행
            row_imgs = []
            for j in range(8):  # 8열
                idx = i * 8 + j
                
                current_img = placeholder.copy()
                
                # 해당 위치의 이미지가 있으면 로드
                if idx < len(sorted_paths) and sorted_paths[idx] and sorted_paths[idx].exists():
                    loaded = cv2.imread(str(sorted_paths[idx]))
                    
                    if loaded is not None:
                        # 크기가 다르면 리사이즈
                        if loaded.shape != (base_h, base_w, base_c):
                            loaded = cv2.resize(loaded, (base_w, base_h))
                        current_img = loaded
                    else:
                        _log_to_status(f"[STITCH] 경고: {sorted_paths[idx].name} 로드 실패.")
                
                row_imgs.append(current_img)

            # 행 단위로 가로로 병합
            all_rows.append(cv2.hconcat(row_imgs))

        # 모든 행을 세로로 병합
        final_img = cv2.vconcat(all_rows)
        cv2.imwrite(str(save_path), final_img)
        _log_to_status(f"[STITCH] 8x4 병합 이미지 저장 완료: {save_path.name}")
        return True
    
    except Exception as e:
        _log_to_status(f"[STITCH] 병합 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


class PCB_Result_LiveStream(BaseClass, FormClass):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
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
        self._setup_file_list_area()

        self.items = []  # 모든 이미지 아이템 리스트
        self.filtered_items = []  # 현재 필터링된 아이템들
        self.thumb_cache = {}  # 썸네일 이미지 캐시
        self.current_filter = "all"  # 현재 적용된 필터
        self.current_search = ""  # 현재 검색어
        self._last_cell_w = None  # 마지막 셀 너비 (폭 변화 없으면 리렌더 생략)  
        
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
            # 정상 이미지만 필터링
            filtered_by_class = [it for it in self.items if it["label"] == "normal"]
        else: 
            # 비정상 이미지만 필터링
            filtered_by_class = [it for it in self.items if it["label"] != "normal"]
            
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

    # 결과 디렉토리에서 이미지를 로드하고 병합하는 함수
    def load_results(self, result_dir_path: str):
        result_dir = Path(result_dir_path)
        if not result_dir.exists():
            QMessageBox.critical(self, "오류", "결과 폴더가 존재하지 않습니다.")
            return

        self._log_to_status(f"[LOAD] 결과 로딩 중... {result_dir.name}")

        self.items.clear()
        self.filtered_items.clear()
        self.thumb_cache.clear()
 
        img_paths = list(result_dir.glob("*.jpg"))
        labels_dir = result_dir / "labels"
        
        # 라벨 파일이 있고 내용이 있으면 비정상으로 판단
        is_abnormal = False
        if labels_dir.exists():
            for txt in labels_dir.glob("*.txt"):
                if txt.stat().st_size > 0:
                    is_abnormal = True
                    break
        
        status = "Abnormal" if is_abnormal else "Normal"

        # 병합된 이미지 경로
        stitched_path = result_dir / "stitched_result.jpg"
        if not stitched_path.exists():
            self._log_to_status("[STITCH] 이미지 병합 시작...")
            _stitch_images_8x4(img_paths, stitched_path)
        
        # 아이템 생성
        item = {
            "path": stitched_path,
            "label": status,
            "labels_dir": labels_dir 
        }
        self.items.append(item)

        # 상태 라벨 업데이트
        if hasattr(self, "map_label"):
            self.map_label.setText(status)
            tcolor = "#C62828" if status == "Abnormal" else "#2E7D32"
            self.map_label.setStyleSheet(f"color: {tcolor}; font-size: 20px; font-weight: bold;")

        self.apply_filter("all")
        self._log_to_status("[LOAD] 완료.")
        
    # 뒤로 가기 함수 
    def go_back(self):
        if self._navigator["go_back"]:
            self._navigator["go_back"]()

    def showEvent(self, e):
        super().showEvent(e)

    def hideEvent(self, e):
        super().hideEvent(e)

    def _process_next_simulation_folder(self):
        if not self.simulation_queue:
            self._log_to_status("[SIM] 모든 항목 처리 완료.")
            if self.sim_timer:
                self.sim_timer.stop()
            return

        folder_name = self.simulation_queue.pop(0)
        self._log_to_status(f"[SIM] 처리 시작: {folder_name}")
        status = "Normal" 
        if "_Abnormal" in folder_name:
            status = "Abnormal"

        source_dir = LIVE_TEST_DIR / folder_name
        stitched_image_name = f"{folder_name}.jpg" 
        stitched_image_path = SIMULATION_CACHE_DIR / stitched_image_name

        if not source_dir.exists():
            self._log_to_status(f"[SIM] 오류: 소스 폴더 없음 {source_dir}")
            return 

        if not stitched_image_path.exists():
            img_paths = [p for p in source_dir.glob("*.jpg") if p.is_file()]
            if not img_paths:
                self._log_to_status(f"[SIM] 오류: {source_dir}에 원본 이미지가 없음")
                return
            self._log_to_status(f"[SIM] {len(img_paths)}개 이미지 병합 시작 -> {stitched_image_name}")
            if not _stitch_images_8x4(img_paths, stitched_image_path):
                self._log_to_status(f"[SIM] 오류: {folder_name} 병합 실패")
                return
        else:
            self._log_to_status(f"[SIM] 캐시된 이미지 사용: {stitched_image_name}")
        item = {
            "path": stitched_image_path,
            "classes": set() if status == "Normal" else {99},
            "label": status 
        }
        self.items.insert(0, item) 
        self._apply_current_filters() 

        if hasattr(self, "map_label"):
             self.map_label.setText(status) 
             tcolor = _label_text_color(status) 
             self.map_label.setStyleSheet(f"color: {tcolor}; font-size: 20px; font-weight: bold;")

        if not self.simulation_queue:
            self._log_to_status("[SIM] 시뮬레이션 완료.")
            if self.sim_timer:
                self.sim_timer.stop()

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

    # 그리드를 재구성하여 이미지 썸네일을 표시하는 함수
    def _rebuild_grid(self, items):
        # 기존 그리드 아이템 제거
        while self._grid.count():
            it = self._grid.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()

        cell_w = self._compute_cell_width()
        self._last_cell_w = cell_w
        r = c = 0  # 그리드 행, 열 인덱스

        for rec in items:
            p = rec["path"]
            pix = self._get_thumb(p, cell_w)  # 썸네일 가져오기

            frame = QFrame()
            frame.setFrameShape(QFrame.StyledPanel)
            v = QVBoxLayout(frame)
            v.setContentsMargins(6, 6, 6, 6)
            
            # 라벨에 따라 배경색 설정
            label_text = rec["label"]
            if label_text == "Normal":
                bg_color = CLASS_COLORS["normal"]
            elif label_text == "Abnormal":
                bg_color = CLASS_COLORS["abnormal"]
            elif "," in label_text:  # 여러 클래스가 있는 경우
                bg_color = "#F5F5F7"
            else:
                bg_color = CLASS_COLORS.get(label_text, "#F8F8F8")
    
            border_color = _darken(bg_color)  # 테두리 색상 (배경색보다 어둡게)
            frame.setStyleSheet(f"""
                QFrame {{
                    background-color: {bg_color};
                    border: 3px solid {border_color};
                    border-radius: 8px;
                }}
            """)

            # 클릭 가능한 이미지 라벨 생성
            img_lbl = ClickableImageLabel()
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setScaledContents(False)
            img_lbl.setPixmap(pix)
            img_lbl.setFixedWidth(cell_w)

            # 이미지 클릭 시 상세 보기 다이얼로그 표시
            def show_popup(path=p, label_text=rec["label"]):
                confidence_data = self._read_confidence_data(
                    path, None 
                )
                dialog = ImagePopupDialog(path, label_text, confidence_data, self)
                dialog.exec_()

            img_lbl.clicked.connect(show_popup)
            img_lbl.setAlignment(Qt.AlignCenter)
            img_lbl.setScaledContents(False)
            img_lbl.setPixmap(pix)
            img_lbl.setFixedWidth(cell_w)

            # 파일명과 라벨 표시
            cap = QLabel(f"{p.name}\n[{rec['label']}]")
            cap.setAlignment(Qt.AlignCenter)
            cap.setWordWrap(True)

            tcolor = _label_text_color(rec["label"])
            cap.setStyleSheet(f"""
                QLabel {{
                    color: {tcolor};
                    background: rgba(255,255,255,0.60);
                    padding: 3px 6px;
                    border-radius: 6px;
                    font-size: 12px;
                    font-weight: 600;
                }}
            """)

            v.addWidget(img_lbl)
            v.addWidget(cap)

            self._grid.addWidget(frame, r, c)
            c += 1
            if c >= GRID_COLS:
                c = 0
                r += 1
                
        self._container.setMinimumWidth(self.scroll_area_images.viewport().width())
        self._container.adjustSize()

    # 썸네일 이미지를 가져오는 함수 (캐시 우선)
    def _get_thumb(self, path: Path, target_w: int) -> QPixmap:
        key = f"{path}:{target_w}"
        if key in self.thumb_cache:
            return self.thumb_cache[key]
        pm = QPixmap(str(path))
        if pm.isNull():
            # 이미지 로드 실패 시 회색 플레이스홀더 생성
            pm = QPixmap(target_w, int(target_w * 0.75))
            pm.fill(Qt.lightGray)
        thumb = pm.scaledToWidth(target_w, Qt.SmoothTransformation)
        self.thumb_cache[key] = thumb  # 캐시에 저장
        return thumb

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
            if label_text == "Normal":
                bg_color = CLASS_COLORS["normal"]
            elif label_text == "Abnormal": 
                bg_color = CLASS_COLORS["abnormal"]
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

# 이미지 상세 보기 다이얼로그 클래스 (자 없이 표시하는 버전)
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
            scaled_pixmap = pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
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

        # Confidence 데이터가 있으면 상세 정보 표시
        if confidence_data:
            text_content = f"파일명: {image_path.name}\n\n검출된 객체:\n"
            for i, (cls_id, conf, bbox) in enumerate(confidence_data, 1):
                cls_name = IDX2NAME.get(cls_id, f"Unknown({cls_id})")
                text_content += f"{i}. 클래스: {cls_name} | 확신도: {conf:.3f} | 위치: {bbox}\n"
        
        elif label_text == "normal":
            text_content = f"파일명: {image_path.name}\n\n검출된 객체: 없음 (Normal)"
            
        else: 
            text_content = f"파일명: {image_path.name}\n\n상태: {label_text.upper()}\n(32개 이미지 종합 결과)"


        confidence_label.setText(text_content)

        layout.addWidget(image_label)
        layout.addWidget(confidence_label)
        
# 가로 자 위젯 클래스 (이미지 위에 cm 단위로 표시)
class HorizontalRuler(QWidget):
    def __init__(self, real_w_cm=8.0, parent=None):
        super().__init__(parent)
        self.real_w_cm = real_w_cm  # 실제 물리적 너비 (cm)
        self.setMinimumHeight(30)
        self.setStyleSheet("background-color: #F0F0F0;")

    def paintEvent(self, event):
        """자 그리기 이벤트 처리"""
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            img_width = self.width() - 15 
            if self.real_w_cm == 0 or img_width <= 0: return 
            
            pixels_per_cm = img_width / self.real_w_cm  # 픽셀당 cm 비율 계산
            
            ruler_y_base = self.height() - 5 
            tick_major_len = 10  # 주요 눈금 길이
            tick_minor_len = 5  # 보조 눈금 길이

            pen = QPen(Qt.black, 1, Qt.SolidLine)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)
            painter.drawLine(0, ruler_y_base, img_width, ruler_y_base)  # 기준선 그리기

            # 주요 눈금 그리기 (1cm 간격)
            for i in range(int(self.real_w_cm) + 1):
                x = int(i * pixels_per_cm)
                painter.drawLine(x, ruler_y_base, x, ruler_y_base - tick_major_len)
                text_rect = QRect(x - 10, ruler_y_base - tick_major_len - 12, 20, 12)
                painter.drawText(text_rect, Qt.AlignCenter, str(i))

            # 보조 눈금 그리기 (0.5cm 간격)
            for i in range(int(self.real_w_cm * 2)):
                if i % 2 == 0: continue
                x = int((i / 2.0) * pixels_per_cm)
                painter.drawLine(x, ruler_y_base, x, ruler_y_base - tick_minor_len)
        except Exception:
            pass 
        painter.end()

# 세로 자 위젯 클래스 (이미지 왼쪽에 cm 단위로 표시)
class VerticalRuler(QWidget):
    def __init__(self, real_h_cm=4.0, parent=None):
        super().__init__(parent)
        self.real_h_cm = real_h_cm  # 실제 물리적 높이 (cm)
        self.setMinimumWidth(30) 
        self.setStyleSheet("background-color: #F0F0F0;")

    def paintEvent(self, event):
        """자 그리기 이벤트 처리"""
        super().paintEvent(event)
        painter = QPainter(self)
        try:
            img_height = self.height() - 15 
            if self.real_h_cm == 0 or img_height <= 0: return 
            
            pixels_per_cm = img_height / self.real_h_cm  # 픽셀당 cm 비율 계산
            
            ruler_x_base = self.width() - 5 
            tick_major_len = 10  # 주요 눈금 길이
            tick_minor_len = 5  # 보조 눈금 길이

            pen = QPen(Qt.black, 1, Qt.SolidLine)
            painter.setPen(pen)
            font = painter.font()
            font.setPointSize(8)
            painter.setFont(font)

            painter.drawLine(ruler_x_base, 0, ruler_x_base, img_height)  # 기준선 그리기

            # 주요 눈금 그리기 (1cm 간격)
            for i in range(int(self.real_h_cm) + 1):
                y = int(i * pixels_per_cm)
                painter.drawLine(ruler_x_base, y, ruler_x_base - tick_major_len, y)
                text_rect = QRect(ruler_x_base - tick_major_len - 22, y - 6, 20, 12)
                painter.drawText(text_rect, Qt.AlignRight | Qt.AlignVCenter, str(i))

            # 보조 눈금 그리기 (0.5cm 간격)
            for i in range(int(self.real_h_cm * 2)):
                if i % 2 == 0: continue
                y = int((i / 2.0) * pixels_per_cm)
                painter.drawLine(ruler_x_base, y, ruler_x_base - tick_minor_len, y)
        except Exception:
            pass 
        painter.end()

# 이미지 상세 보기 다이얼로그 클래스 (자 포함 버전)
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
 
        pixmap = QPixmap(str(image_path))
        scaled_pixmap = pixmap.scaled(900, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setPixmap(scaled_pixmap)
        image_label.setFixedSize(scaled_pixmap.size()) 

        # 가로 자 생성 (8cm 기준)
        top_ruler = HorizontalRuler(real_w_cm=8.0)
        top_ruler.setFixedSize(scaled_pixmap.width() + 15, 30) 

        # 세로 자 생성 (4cm 기준)
        left_ruler = VerticalRuler(real_h_cm=4.0)
        left_ruler.setFixedSize(30, scaled_pixmap.height() + 15) 

        # 자 교차점 코너 위젯
        corner = QWidget()
        corner.setFixedSize(30, 30)
        corner.setStyleSheet("background-color: #F0F0F0;")

        # 자와 이미지를 그리드로 배치
        ruler_grid_layout = QGridLayout()
        ruler_grid_layout.setSpacing(0) 
        ruler_grid_layout.setContentsMargins(0, 0, 0, 0)
        ruler_grid_layout.addWidget(corner,       0, 0)  # 좌상단 코너
        ruler_grid_layout.addWidget(top_ruler,    0, 1)  # 상단 가로 자
        ruler_grid_layout.addWidget(left_ruler,   1, 0)  # 왼쪽 세로 자
        ruler_grid_layout.addWidget(image_label,  1, 1)  # 이미지
        ruler_container = QWidget()
        ruler_container.setLayout(ruler_grid_layout)
        ruler_container.setFixedSize(
            30 + scaled_pixmap.width() + 15, 
            30 + scaled_pixmap.height() + 15
        )

        layout.addWidget(ruler_container, 0, Qt.AlignCenter)
        
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

        # Confidence 데이터가 있으면 상세 정보 표시
        if confidence_data:
            text_content = f"파일명: {image_path.name}\n\n검출된 객체:\n"
            for i, (cls_id, conf, bbox) in enumerate(confidence_data, 1):
                cls_name = IDX2NAME.get(cls_id, f"Unknown({cls_id})")
                text_content += f"{i}. 클래스: {cls_name} | 확신도: {conf:.3f} | 위치: {bbox}\n"
        
        elif label_text == "Normal":
            text_content = f"파일명: {image_path.name}\n\n검출된 객체: 없음 (Normal)"
            
        else:
            text_content = f"파일명: {image_path.name}\n\n상태: {label_text.upper()}\n(32개 이미지 종합 결과)"

        confidence_label.setText(text_content)
        layout.addWidget(confidence_label)

# 클릭 가능한 이미지 라벨 클래스
class ClickableImageLabel(QLabel):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.PointingHandCursor)  # 손가락 커서로 변경
    
    def mousePressEvent(self, event):
        """마우스 클릭 이벤트 처리"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)
        
def _log_to_status(msg: str):
    """로그 메시지를 출력하는 함수"""
    print(msg, flush=True)
