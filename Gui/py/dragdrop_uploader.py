# dragdrop_uploader.py
# -*- coding: utf-8 -*-
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QFrame
import os

def _collect_files_from_dir(dir_path, accept_exts=None, recursive=True):
    results = []
    if accept_exts:
        accept_exts = {ext.lower() for ext in accept_exts}
    for root, dirs, files in os.walk(dir_path):
        for f in files:
            full = os.path.join(root, f)
            if accept_exts:
                _, ext = os.path.splitext(full)
                if ext.lower() not in accept_exts:
                    continue
            results.append(os.path.normpath(full))
        if not recursive:
            break
    return results

class FileDropWidget(QFrame):
    filesDropped = pyqtSignal(list)

    def __init__(self, text="여기로 드래그하여 파일 업로드", 
                 accept_exts=None, accept_dirs=True, recursive=True, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.accept_exts = {ext.lower() for ext in accept_exts} if accept_exts else None
        self.accept_dirs = accept_dirs
        self.recursive = recursive

        self.setObjectName("FileDropWidget")
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

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.label = QLabel(text, self)
        self.label.setObjectName("DropLabel")
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setProperty("dragActive", False)
        self.setMinimumHeight(120) 

    def dragEnterEvent(self, event):
        def _url_allowed(url):
            p = url.toLocalFile()
            if not p:
                return False
            if os.path.isdir(p):
                return self.accept_dirs 
            if self.accept_exts:
                _, ext = os.path.splitext(p)
                return ext.lower() in self.accept_exts
            return True  

        if event.mimeData().hasUrls() and any(_url_allowed(u) for u in event.mimeData().urls()):
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self); self.style().polish(self); self.update()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self); self.update()
        super().dragLeaveEvent(event)

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
            path = os.path.normpath(path)

            if os.path.isdir(path):
                if self.accept_dirs:
                    collected.extend(_collect_files_from_dir(
                        path, accept_exts=self.accept_exts, recursive=self.recursive
                    ))
                continue

            if self.accept_exts:
                _, ext = os.path.splitext(path)
                if ext.lower() not in self.accept_exts:
                    continue

            collected.append(path)
            
        deduped = sorted(set(collected))

        if deduped:
            self.filesDropped.emit(deduped)
            event.acceptProposedAction()
        else:
            event.ignore()

        self.setProperty("dragActive", False)
        self.style().unpolish(self); self.style().polish(self); self.update()


    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.acceptProposedAction()
        else:
            event.ignore()
