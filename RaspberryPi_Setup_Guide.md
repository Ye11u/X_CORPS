# ORAIL 인수인계 자료 – Raspberry Pi & Camera Setup

## 1. 개요
본 문서는 ORAIL 연구실 IN-GPS / X-CORPS 프로젝트에서 사용한  
Raspberry Pi 및 카메라 장비에 대한 인수인계 자료이다.

## 2. 라즈베리파이 장비 구성
### 2.1 장비 정보

- **제품명**: Raspberry Pi 5 (8GB)
- **CPU**: BCM2712 Quad-Core 2.4GHz
- **RAM**: 8GB
- **저장매체**: microSD 카드  
  (현재 256GB 삼성 제품 사용 중, OS는 microSD에 설치)
- **비디오 출력**: micro-HDMI ×2
- **USB 포트**: USB 3.0 ×2, USB 2.0 ×2
- **네트워크**: Gigabit Ethernet, Wi-Fi, Bluetooth
- **카메라 포트**: CSI (MIPI 4-lane)
- **전원**: USB-C 5V 5A  
구매처링크: https://www.eleparts.co.kr/goods/view?no=13323153

※ 장비는 이미 구비되어 있으므로, 교체 또는 추가 구매 시 참고용으로만 활용하면 된다.
![라즈베리파이 장비 구성]
![인수인계_ORAIL_page-0003](https://github.com/user-attachments/assets/aea8f433-19a4-4e06-83fb-b549f05fcb3c)


## 3. 라즈베리파이 초기 Setup
### 3.1 Raspberry Pi OS 설치

1. Raspberry Pi Imager 설치
2. Raspberry Pi OS 선택
3. microSD 카드 선택
4. Write 실행  
참고: https://m.blog.naver.com/icbanq/223382909813  
https://makeutil.tistory.com/253

※ 현재 장비에는 OS가 이미 설치되어 있으므로 새로 설치할 필요는 없다.  
OS 재설치는 포맷과 동일하므로, 기존 코드 및 환경이 삭제될 수 있다.

![인수인계_ORAIL_page-0004](https://github.com/user-attachments/assets/fe8bf779-0e5e-43e7-ab10-23ec581dacc5)


## 4. 원격 접속 환경 구성
라즈베리파이는 실제로는 하나의 소형 컴퓨터이므로  
모니터를 직접 연결하거나, 노트북에서 원격 접속하여 사용할 수 있다.

실험 및 개발 과정에서는 **원격 접속 방식 사용이 필수**이다.
- 모니터, 키보드, 마우스를 라즈베리파이에 직접 연결
- Wi-Fi 설정 진행
- 개인 핫스팟 사용 권장 (학교 Wi-Fi는 불안정)
- 연결 후 IP 주소를 반드시 기록
- 노트북에 Putty 또는 VNC Viewer 설치
- 라즈베리파이 IP 주소로 접속
- 이후 모든 작업은 노트북에서 진행
  
링크 : https://velog.io/@easyhyun00/%EB%9D%BC%EC%A6%88%EB%B2%A0%EB%A6%AC%ED%8C%8C%EC%9D%B4-OS-%EC%84%A4%EC%B9%98-%EC%9B%90%EA%B2%A9-%EC%A0%91%EC%86%8DVNC-Viewer

## 5. 카메라 장비 구성
### 5.1 사용 장비

- Raspberry Pi HQ Camera Module
- 줌 렌즈 (사용)
- 삼각대
  
카메라모듈https://www.eleparts.co.kr/EPXTPVJH  
망원렌즈https://www.eleparts.co.kr/goods/view?no=9544398  
줌렌즈https://www.eleparts.co.kr/goods/view?no=9604451  
삼각대https://www.eleparts.co.kr/goods/view?no=9472997  

※ 망원렌즈도 구매했으나, 줌 렌즈를 주로 사용하였다.

![인수인계_ORAIL_page-0005](https://github.com/user-attachments/assets/efdd05a3-7d8c-40f5-8555-e5d0a365aea2)

## 6. 카메라–라즈베리파이 연결
### 6.2 소프트웨어 확인

아래 명령어 실행 시 카메라 화면이 출력되면 정상 연결이다.

rpicam-hello -t 0

![인수인계_ORAIL_page-0006](https://github.com/user-attachments/assets/98789ca8-1127-461f-8eb1-a2b801af4c6f)

## 7. 데이터셋 제작 방법
1. PCB 촬영
   - 구도, 조명, 색상이 매우 중요
   - 최대한 동일한 환경에서 촬영할 것

2. 누끼 제거
   - PCB 외 배경 제거
   - https://www.photoroom.com/ko/tools/background-remover

3. 이미지 분할
   - PCB 이미지를 32등분
   - 코드로 자동 분할 가능
   
![인수인계_ORAIL_page-0009](https://github.com/user-attachments/assets/0abcd5be-f816-4cbe-ab2a-c2c8ebd887f0)

## 8. 세팅 참고용 사진

![인수인계_ORAIL_page-0007](https://github.com/user-attachments/assets/3b4c16b8-6620-4993-9ddf-efdab9606c34)

![인수인계_ORAIL_page-0008](https://github.com/user-attachments/assets/e2e8837e-cd0e-4d68-ad12-b358c79364c3)

## 9. 기타 유의사항

![인수인계_ORAIL_page-0010](https://github.com/user-attachments/assets/2224ab8e-fd89-4d54-8660-bd5a45d45611)





