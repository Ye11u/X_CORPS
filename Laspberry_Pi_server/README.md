# Server 연결 방법 

동일 네트워크에서 라즈베리파이(송신) -> GUI(수신)
라즈베리파이에 키보드, 모니터를 연결하지 않고 노트북과 랜선을 이용하여 접속

## 준비물 

- 멀티탭 (1~2구)
- LAN 케이블
- LAN-to-C ot LAN-to-USB 변환 케이블
- 라즈베리파이 전원 장치
- 카메라, 카메라 케이블

## 작동 방법 
 1. 라즈베리파이와 노트북을 LAN 케이블로 연결 (LAN-to-C or LAN-to-USB 케이블 이용)
 2. 노트북 cmd 창 열기
    ```bash
    ssh pi@raspberrypi.local //입력
    ```
    (1) 만약 연결이 안 된 상태라면 아래와 같은 결과가 출력됨 
    <img width="1307" height="85" alt="image" src="https://github.com/user-attachments/assets/586c7123-236c-43ba-9d80-71dc97f017f3" />
    
      - 노트북과 케이블을 뺐다 다시 꼽아보기
      - 윈도우/네트워크 연결 보기 메뉴에서 “식별 중..” 상태인지 확인 => “식별되지 않은 네트워크"로 뜨면 연결된 것
      - 위 2가지 방법을 실행한 뒤, 다시 입력

    (2) 연결된 상태라면 아래와 같은 결과가 출력됨
    <img width="1307" height="307" alt="image" src="https://github.com/user-attachments/assets/5788a5ac-854c-4b38-8811-0622f18d08d9" />

    (3) VNC Viewer 열고 raspberrypi.local 입력 & 접속
    
    (4) 핫스팟 연결
      - 노트북 핫스팟 연결 
      -  VNC Viewer 접속해서 라즈베리파이도 노트북과 같은 핫스팟 연결

    (5) 라즈베리파이 VNC Viewer : code .로 vs code를 열고, home/GPS 디렉토리 폴더를 선택
    
    (6) 라즈베리파이 VNC Viewer : vs code의 터미널 창에 python server.py 실행 (서버 돌아감)

    (7) 노트북: GUI 실행
      - 만약 실시간 카메라 화면이 안 보인다면, VNC Viewer : vs code의 터미널 창에 뜬 서버 주소와 노트북 GUI에 있는 주소가 같은 지 확인  
