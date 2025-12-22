#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPS_GUI 프로젝트를 단일 EXE로 빌드하는 스크립트 (간소화된 버전)
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_basic_requirements():
    """기본 요구사항만 확인"""
    print("=== 기본 요구사항 확인 ===")
    
    # Python 버전 확인
    print(f"Python 버전: {sys.version}")
    
    # 핵심 패키지만 확인
    core_packages = {
        'PyQt5': 'PyQt5', 
        'numpy': 'numpy',
        'opencv-python': 'cv2',
    }
    
    missing_packages = []
    for package_name, import_name in core_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} 설치됨")
        except ImportError:
            missing_packages.append(package_name)
            print(f"✗ {package_name} 누락")
    
    if missing_packages:
        print(f"\n누락된 패키지들을 설치하세요:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✓ 핵심 패키지들이 설치되어 있습니다")
    return True

def check_project_structure():
    """프로젝트 구조 확인"""
    print("\n=== 프로젝트 구조 확인 ===")
    
    required_paths = {
        'py/login.py': '진입점 파일',
        'ui/': 'UI 파일 디렉토리',
        'GPS_GUI.spec': 'PyInstaller spec 파일'
    }
    
    all_good = True
    for path, description in required_paths.items():
        if os.path.exists(path):
            print(f"✓ {path} ({description})")
        else:
            print(f"✗ {path} 누락 ({description})")
            all_good = False
    
    return all_good

def clean_previous_build():
    """이전 빌드 결과 정리"""
    print("\n=== 이전 빌드 결과 정리 ===")
    
    paths_to_clean = ['build', 'dist', '__pycache__']
    
    for path in paths_to_clean:
        if os.path.exists(path):
            if os.path.isdir(path):
                shutil.rmtree(path)
                print(f"✓ 디렉토리 삭제: {path}")
            else:
                os.remove(path)
                print(f"✓ 파일 삭제: {path}")

def build_with_spec():
    """spec 파일을 사용해서 빌드"""
    print("\n=== PyInstaller 빌드 시작 ===")
    
    spec_file = 'GPS_GUI.spec'
    if not os.path.exists(spec_file):
        print(f"✗ {spec_file} 파일이 없습니다!")
        return False
    
    try:
        # PyInstaller 실행
        cmd = [sys.executable, '-m', 'PyInstaller', '--clean', spec_file]
        print(f"실행 명령어: {' '.join(cmd)}")
        print("빌드 진행 중... (시간이 걸릴 수 있습니다)")
        
        # 실시간 출력을 위해 subprocess.Popen 사용
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            universal_newlines=True
        )
        
        # 실시간 출력
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        return_code = process.poll()
        
        if return_code == 0:
            print("\n✓ 빌드 성공!")
            return True
        else:
            print(f"\n✗ 빌드 실패! (종료 코드: {return_code})")
            return False
            
    except Exception as e:
        print(f"빌드 중 오류 발생: {e}")
        return False

def verify_build():
    """빌드 결과 확인"""
    print("\n=== 빌드 결과 확인 ===")
    
    # onefile 방식: dist/GPS_GUI.exe (단일 파일)
    exe_path = os.path.join('dist', 'GPS_GUI.exe')
    
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024 * 1024)
        print(f"✓ 단일 실행 파일 생성됨: {exe_path}")
        print(f"  파일 크기: {size_mb:.1f} MB")
        return True
    else:
        print("✗ 단일 실행 파일을 찾을 수 없습니다")
        
        # dist 폴더 내용 확인
        if os.path.exists('dist'):
            dist_files = os.listdir('dist')
            print(f"dist 폴더 내용: {dist_files}")
        
        return False

def main():
    """메인 빌드 프로세스"""
    print("🚀 GPS_GUI 단일 EXE 빌드 스크립트 시작")
    print("=" * 50)
    
    # 1. 기본 요구사항 확인
    if not check_basic_requirements():
        print("\n❌ 기본 요구사항을 만족하지 않습니다.")
        return False
    
    # 2. 프로젝트 구조 확인
    if not check_project_structure():
        print("\n❌ 프로젝트 구조에 문제가 있습니다.")
        return False
    
    # 3. 이전 빌드 정리
    clean_previous_build()
    
    # 4. 빌드 실행
    if not build_with_spec():
        print("\n❌ 빌드에 실패했습니다.")
        print("\n💡 문제 해결 방법:")
        print("1. 가상환경이 활성화되어 있는지 확인")
        print("2. pip install PyInstaller 재실행")
        print("3. spec 파일의 경로들이 올바른지 확인")
        return False
    
    # 5. 결과 확인
    if not verify_build():
        print("\n❌ 빌드 결과 확인에 실패했습니다.")
        return False
    
    print("\n" + "=" * 50)
    print("🎉 단일 EXE 빌드 완료!")
    print(f"📁 생성된 파일: dist/GPS_GUI.exe")
    print("\n📋 다음 단계:")
    print("1. cd dist && GPS_GUI.exe 로 테스트 실행")
    print("2. 정상 작동 확인 후 배포")
    
    return True

if __name__ == '__main__':
    success = main()
    input("\n아무 키나 누르세요...")  # 실행 결과 확인을 위해
    sys.exit(0 if success else 1)