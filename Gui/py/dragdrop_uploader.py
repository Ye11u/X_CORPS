#######################################################################################
# 드래그 앤 드롭 파일 업로더 >> 파일/폴더를 드래그하여 업로드할 수 있는 위젯
#######################################################################################
# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
import os

def _collect_files_from_dir(dir_path, accept_exts=None, recursive=True):
    """디렉토리에서 파일을 수집하는 함수 (재귀적 탐색 지원)"""
    results = []
    if accept_exts:
        # 확장자를 소문자로 변환하여 비교
        accept_exts = {ext.lower() for ext in accept_exts}
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            full = os.path.join(root, f)
            if accept_exts:
                # 허용된 확장자만 수집
                _, ext = os.path.splitext(full)
                if ext.lower() not in accept_exts:
                    continue
            results.append(os.path.normpath(full))
        if not recursive:
            # 재귀적 탐색이 비활성화되어 있으면 첫 번째 레벨만 탐색
            break
    return results

class FileDropWidget(QFrame):
    """파일 드래그 앤 드롭을 지원하는 위젯 클래스"""
    filesDropped = pyqtSignal(list)  # 파일 드롭 완료 시그널 (파일 경로 리스트 전달)

    def __init__(self, text="여기로 드래그하여 파일 업로드", 
                 accept_exts=None, accept_dirs=True, recursive=True, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)  # 드롭 이벤트 수신 활성화
        self.accept_exts = {ext.lower() for ext in accept_exts} if accept_exts else None  # 허용할 파일 확장자 (소문자로 변환)
        self.accept_dirs = accept_dirs  # 디렉토리 드롭 허용 여부
        self.recursive = recursive  # 디렉토리 내부 재귀적 탐색 여부

        self.setObjectName("FileDropWidget")
        # 위젯 스타일 설정 (드래그 상태에 따라 시각적 피드백 제공)
        self.setStyleSheet("""
            /* 기본: 하늘 */
            QFrame#FileDropWidget {
                border: 3px dashed #4DA3FF;      /* 중간 하늘 */
                border-radius: 16px;
                background: #EAF6FF;             /* 밝은 하늘 */
            }

            /* 마우스 넘어가면: 좀 더 진하게 */
            QFrame#FileDropWidget:hover {
                border-color: #1C8CFF;           /* 진한 파랑 */
                background: #DFF1FF;             /* 좀 더 선명 */
            }

            /* 드래그 중이면: 실선 + 그라디언트 */
            QFrame#FileDropWidget[dragActive="true"] {
                border: 3px solid #1C8CFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #EAF6FF,              /* 위쪽 연한 하늘 */
                    stop:1 #CFE9FF               /* 아래쪽 파랑 */
                );
            }

            QLabel#DropLabel {
                color: #0E4775;                  /* 진한 파랑*/
                font-size: 16px;
                font-weight: 600;
                padding: 8px 4px;
            }

            /* 비활성화 */
            QFrame#FileDropWidget:disabled {
                border-color: #BFD9F5;
                background: #F3F9FF;
            }
        """)

        # 레이아웃 및 라벨 설정
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.label = QLabel(text, self)
        self.label.setObjectName("DropLabel")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setAttribute(Qt.WA_StyledBackground, True)  # 스타일시트 배경 적용
        self.setProperty("dragActive", False)  # 드래그 활성 상태 초기화
        self.setMinimumHeight(120)  # 최소 높이 설정 

    # 드래그가 위젯 영역에 진입했을 때 호출되는 함수
    def dragEnterEvent(self, event):
        def _url_allowed(url):
            """URL이 허용되는지 확인하는 내부 함수"""
            p = url.toLocalFile()
            if not p:
                return False
            if os.path.isdir(p):
                # 디렉토리인 경우 accept_dirs 설정 확인
                return self.accept_dirs 
            if self.accept_exts:
                # 파일인 경우 확장자 확인
                _, ext = os.path.splitext(p)
                return ext.lower() in self.accept_exts
            return True  # 확장자 제한이 없으면 모두 허용

        # 허용되는 파일/폴더가 하나라도 있으면 드롭 허용
        if event.mimeData().hasUrls() and any(_url_allowed(u) for u in event.mimeData().urls()):
            event.setDropAction(Qt.CopyAction)  # 복사 액션으로 설정
            event.acceptProposedAction()
            self.setProperty("dragActive", True)  # 드래그 활성 상태로 변경
            self.style().unpolish(self); self.style().polish(self); self.update()  # 스타일 업데이트
        else:
            event.ignore()

    # 드래그가 위젯 영역을 벗어났을 때 호출되는 함수
    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)  # 드래그 활성 상태 해제
        self.style().unpolish(self); self.style().polish(self); self.update()  # 스타일 업데이트
        super().dragLeaveEvent(event)

    # 파일이 위젯에 드롭되었을 때 호출되는 함수
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls:
            event.ignore()
            self.setProperty("dragActive", False)
            self.style().unpolish(self); self.style().polish(self); self.update()
            return

        collected = []
        for url in urls:
            path = url.toLocalFile()
            if not path:
                continue
            path = os.path.normpath(path)  # 경로 정규화

            # 디렉토리인 경우 내부 파일 수집
            if os.path.isdir(path):
                if self.accept_dirs:
                    collected.extend(_collect_files_from_dir(
                        path, accept_exts=self.accept_exts, recursive=self.recursive
                    ))
                continue

            # 파일인 경우 확장자 확인
            if self.accept_exts:
                _, ext = os.path.splitext(path)
                if ext.lower() not in self.accept_exts:
                    continue

            collected.append(path)
            
        # 중복 제거 및 정렬
        deduped = sorted(set(collected))

        # 수집된 파일이 있으면 시그널 발생
        if deduped:
            self.filesDropped.emit(deduped)
            event.acceptProposedAction()
        else:
            event.ignore()

        self.setProperty("dragActive", False)  # 드래그 활성 상태 해제
        self.style().unpolish(self); self.style().polish(self); self.update()  # 스타일 업데이트


    # 드래그 중 위젯 위에서 이동할 때 호출되는 함수
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)  # 복사 액션으로 설정
            event.acceptProposedAction()
        else:
            event.ignore()
