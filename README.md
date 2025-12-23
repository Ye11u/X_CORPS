# IN-GPS & X-CORPS (ORAIL_TEAM)

## 개요

이 코드는 반도체 제조 산업 불량 탐지 task에 대해 pcb dataset에서의 결함 클래스 분류 및 결함 부위 표시 기능을 구현한 어플리케이션이다.
자체 pcb dataset을 제작하여 모델의 성능을 검증하였다. 

- **Gui 폴더: 데스크탑 Gui 어플리케이션. README를 참고.**
- **Raspberry_Pi_server 폴더: 라즈베리파이 카메라 서버. README를 참고.**
- **Model 폴더: Ai 모델의 훈련, 추론, 증강 방법을 정리함**

## 시연 영상

(영상첨부)

## 데이터셋 설명 

이 코드는 자체적으로 제작된 pcb dataset을 사용하여 검증되었다. PCB는 반도체 칩과 같은 전자 부붐을 고정하고, 연결하는 기판이다. 
자체 제작한 pcb dataset의 정상, 이상에 대한 샘플이 아래 그림에 나타나있다. 

<img width="1142" height="567" alt="pcb_dataset" src="https://github.com/user-attachments/assets/27756af0-ec73-469c-8d74-e9dfa1c9dc71" />

실제 PCB 산업 현장에서 발생할 수 있는 결함은 short, open, pad open, silk, 총 4가지 결함이다.
결함의 패턴이 고정적이기 때문에 우리는 데이터를 정상, shrot, open, pad open, silk 총 5가지 클래스로 분류하였다. 

<img width="1066" height="331" alt="pcb 결함" src="https://github.com/user-attachments/assets/0ea849e1-bc02-4db8-af94-8eb5c1671ad3" />
 
- short 결함: 트레이스가 서로 붙어버린 결함
- open 결함: 트레이스가 끊여져 전류가 흐를 수 없는 결함 
- pad open 결함: 패드에 정상적으로 납땜이 되지 않은 결함 
- silk 결함: silk 끼리 겹치거나 잘못된 위치에 프린팅된 결함
  
## 주요 기능

### 결함 부위 탐지 및 분류 
- 사용자가 학습 데이터로 이상 이미지와 그에 대한 바운딩 박스 ground truth를 가지고 있을 경우 사용하는 기능이다.
- yolov8n 모델을 사용하여 결함 위치에 바운딩 박스 표시 및 결함 종류 분류를 학습하는 기능이다.

### 실시간 결함 탐지 
- 라즈베리파이 카메라를 이용하여 실시간 PCB를 송출받고, 실시간으로 결함을 탐지하는 기능이다.

### 실시간 학습 모니터링
- 학습 중인 모델의 loss를 실시간으로 확인할 수 있다.
- 학습이 완료되면 모델의 mAP 50이 표시된다.

## 설치 방법

이 프로젝트를 실행하기 위해 필요한 라이브러리는 다음과 같다. 

1. 노트북, 데스크탑 환경에서 실행
GUI
- Python 3.8+
- PyQt5
- PyTorch
- scikit-learn
- OpenCV
- matplotlib
- numpy

Model
- Python 3.8+
- Pytorch
- YAML
- OpenCV
- ultralytics
- pathlib
- zipfile

2. 라즈베리파이 환경에서 실행
Server
- Flask
- Picamera
- OpenCV
- numpy


