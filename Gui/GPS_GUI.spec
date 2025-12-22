block_cipher = None

import sys
import os
import glob
from PyInstaller.compat import is_win

project_root = os.getcwd()

# =========================================================================
# PyTorch DLL 수집 - OnFile 호환 버전
# =========================================================================
my_binaries = []

# 프로젝트 루트의 DLL
dll_files = glob.glob(os.path.join(project_root, '*.dll'))
for dll_path in dll_files:
    my_binaries.append((dll_path, '.'))

# PyTorch lib 디렉토리 - 매우 중요!
try:
    import torch
    torch_path = os.path.dirname(torch.__file__)
    torch_lib_path = os.path.join(torch_path, 'lib')
    
    if os.path.exists(torch_lib_path):
        # 모든 DLL을 루트에 복사
        dll_files_torch = glob.glob(os.path.join(torch_lib_path, '*.dll'))
        for dll_path in dll_files_torch:
            my_binaries.append((dll_path, '.'))
            print(f"[ADDED] {os.path.basename(dll_path)}")
except Exception as e:
    print(f"[WARNING] {e}")

# =========================================================================
# 데이터 파일
# =========================================================================
my_datas = [
    ('ui', 'ui'),                   
    ('inference_model', 'inference_model'),
]

a = Analysis(
    ['py/login.py'],
    pathex=[
        project_root, 
        os.path.join(project_root, 'py'),
        os.path.join(project_root, 'inference_model'),
    ],
    binaries=my_binaries,
    datas=my_datas,
    hiddenimports=[
        # ===== PyQt5 =====
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'PyQt5.QtPrintSupport',
        'PyQt5.sip',
        
        # ===== NumPy, OpenCV =====
        'numpy',
        'numpy.core',
        'numpy.random',
        'cv2',
        
        # ===== PyTorch (매우 중요!) =====
        'torch',
        'torch._C',
        'torch.utils',
        'torch.utils.data',
        'torch.nn',
        'torch.nn.functional',
        'torch.optim',
        'torch.backends',
        'torch.backends.cudnn',
        'torch.multiprocessing',
        'torchvision',
        'torchvision.models',
        'torchvision.transforms',
        
        # ===== 프로젝트 모듈 =====
        'PCB_select_train_or_test',
        'PCB_training_file_upload',
        'PCB_training_monitoring',
        'PCB_test_file_upload',
        'PCB_test_or_live',
        'PCB_live_streaming',
        'PCB_result_live_stream',
        'PCB_result',
        'inference_thread',
        'inference_model.test',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=True,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='GPS_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,  # 명시적으로 onefile 지정
)