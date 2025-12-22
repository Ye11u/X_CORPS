# GUI 기능 및 코드 설명

## 코드개요 

훈련, 테스트, 실시간 테스트 기능을 담은 GUI 어플리케이션 코드입니다. 
파이썬 코드는 ./py 아래 위치하고, 주석으로 기능에 대한 설명이 첨부되어 있습니다.

## 화면 연결

전체적인 워크플로우는 다음과 같습니다. 
<img width="1085" height="720" alt="슬라이드1" src="https://github.com/user-attachments/assets/f4f50c8c-3e72-4b1b-bbc6-1a7cf83730f9" />

### 1. login.py
로그인 화면

<img width="1085" height="720" alt="슬라이드2" src="https://github.com/user-attachments/assets/c999b55f-9f36-4cff-869a-4fdf4bef5a46" />

- 아이디 & 비번: login.py의 user_info 딕셔너리에 정의되어 있습니다. (id: 1, pw: 1)
- login_btn: id와 pw를 입력하고 login_btn을 누르면 다음 화면으로 넘어갑니다.

### 2. PCB_train_or_test.py
(훈련) / (테스트, 실시간 추론)를 선택하는 분기 화면

<img width="1085" height="720" alt="슬라이드3" src="https://github.com/user-attachments/assets/6f33b391-4a2f-441b-a630-10d63140f11e" />

- start_btn_2: 훈련 파일 업로드 화면으로 넘어갑니다.
- start_btn_3: 테스트/실시간 탐지 선택 화면으로 넘어갑니다.
- reset_btn_3: 전체 초기화시킨 뒤, 현재 화면으로 돌아오는 버튼 (ex: 추론 결과 화면에서 홈화면으로 돌아올 때 사용합니다.) 
- home_btn_3: 현재 화면(PCB_train_or_test.ui)로 돌아오는 버튼
- (설명하지 않은 나머지 버튼들은 기능이 없는 버튼입니다.) 

### 3. PCB_training_file_upload.py
훈련시킬 데이터셋(.zip)과 모델 파일(.py)을 업로드하는 화면

<img width="1085" height="720" alt="슬라이드4" src="https://github.com/user-attachments/assets/cafd7a91-61b8-49d6-afc4-6fdfa213ae4c" />

- file_btn: 업로드 하는 파일은 train, test, val 하위 폴더로 이루어진 zip 파일이어야 합니다. (하위 폴더에는 images/labels 폴더로 나뉘어져야 합니다.)
- file_btn_2: 모델 구조 python 파일을 받는 버튼. (실제 사용자로부터 모델 python 코드를 받을 경우 오류가 발생할 수 있기 때문에, 실제 동작하는 모델은 yolov8n.pt로 해당 코드 파일에 내장되어 있습니다.) 
- back_btn: 이전 선택 화면으로 돌아가는 버튼
- next_btn: 모니터링 페이지로 넘어가는 버튼. 버튼을 누르면 next_page()에서 _run_train_py_now()를 호출하며 학습 시작
- FileDropWidget: dragdrop_uploader.py에서 import해서 사용. 파일을 드래그 앤 드랍으로 추가, 추가되었을 때 색상 변경 기능을 구현한 클래스

### 4. PCB_training_monitoring.py
훈련 과정을 모니터링하는 화면

<img width="1047" height="695" alt="image" src="https://github.com/user-attachments/assets/ca9a6a16-4e15-4386-a7ad-70492bd83d71" />

- test_btn: 훈련이 완료되면, test 위해 다음 화면으로 넘어가는 페이지입니다.
- back_btn: 이전 화면으로 돌아가는 버튼입니다. 넘어가지 않는다면, 왼쪽 프레임의 reset_btn을 누르면 됩니다.
- progressBar: 훈련 진행상황을 보여주는 bar
- widget: 자동으로 보여지는 창입니다. 훈련 중 측정되는 6가지 train/val loss(box, cls, dfl loss)와 validation set에 대한 metrics(precision, recall, mAP50, mAP90-95) 그래프를 보여주는 영역입니다.

### 5. PCB_test_or_live.py
(테스트) / (실시간 추론)를 선택하는 분기 화면

<img width="1085" height="720" alt="슬라이드6" src="https://github.com/user-attachments/assets/e4c8880a-6452-432f-b2c9-aa23abcf1d65" />

- start_btn_2: 실시간 추론 화면으로 이동하는 버튼
- start_btn_3: 테스트 화면으로 이동하는 버튼
- back_btn: 이전(훈련/테스트) 화면으로 이동하는 버튼 

### 6. PCB_test_file_upload.py
테스트할 데이터셋(.zip)과 모델 가중치 파일(.pt)을 업로드하는 화면

<img width="1085" height="720" alt="슬라이드7" src="https://github.com/user-attachments/assets/fe241b12-5647-4d00-9408-964f907c558e" />

- file_btn: 업로드 하는 파일은 test 폴더로 이루어진 zip 파일이어야 합니다. (하위 폴더에는 images/labels 폴더로 나뉘어져야 합니다.)
- file_btn_2: 모델 가중치 파일을 받는 버튼. (실제 사용자로부터 모델 가중치를 받을 경우 구조가 맞지 않는 오류가 발생할 수 있기 때문에, 실제 적용되는 가중치 파일은 파일에 내장되어 있습니다.) 
- 추론을 위해 내장된 코드 및 가중치 파일은 PCB_test_file_upload.py 파일의 global 변수로 선언되어 있습니다. (23~35줄)
- 실제 동작하는 추론 코드: Gui\inference_model\test.py
- 내장된 모델 가중치: Gui\inference_model\content\weights\last.pt\
- back_btn: (테스트)/(실시간 추론) 선택 화면으로 돌아가는 버튼
- next_btn: 테스트 결과 화면으로 넘어가는 버튼

### 7. PCB_result.py
테스트 결과를 보여주는 화면

<img width="1085" height="720" alt="슬라이드8" src="https://github.com/user-attachments/assets/8880be15-5700-44ff-b906-b86bff237f12" />

- progressBar: 진행상황을 보여주는 bar
- search_edit: 특정 파일의 결과를 검색을 통해 찾을 수 있음 (_on_search_changed(), _apply_current_filters()에 구현됨)
- btns: 각 클래스로 분류된 파일들을 모아 보여주는 필터 버튼입니다. (apply_filter()의 인자로 클래스 이름을 받도록 구현됨)
- scroll_area_images: 결과 이미지를 보여주는 화면, 필터 및 검색 창 결과가 적용됨
- scroll_area_list: 파일명을 보여주는 화면, 필터 및 검색 창 결과가 적용됨
- 결과를 확인한 뒤, 홈화면(훈련/테스트 선택 화면)으로 돌아가고 싶다면, 왼쪽 프레임의 reset_btn을 누르면 됩니다.

### 8. PCB_live_streaming.py
카메라가 보는 장면을 실시간으로 송출하는 화면

<img width="1085" height="720" alt="슬라이드9" src="https://github.com/user-attachments/assets/e4dc908a-a69a-46dd-938b-77a578275e69" />

- **중요!!!: 라즈베리 파이 서버 주소를 PCB_live_streaming.py의 187줄 url="{여기에 입력}" && PCB_train_or_test.py의 57번째 줄 url="{여기에 입력}"에 넣어주어야 합니다.**
- videa_label: 카메라에 보이는 화면을 gui에서 송출하는 label
- back_btn: (테스트)/(실시간 추론) 선택 화면으로 돌아가는 버튼
- next_btn: 현재 카메라에 보이는 객체를 찍어(라즈베리파이 서버에 요청) 4x8=32 조각으로 나눈 뒤, 추론를 동작시키는 버튼. 누르면 결과 화면으로 넘어갑니다.
- 라즈베리 파잉 서버에서 전달받은 사진은 Gui\inference_model\content\pcb_std\images\test_live_streaming 의 하위 폴더에 timestamp 형식으로 저장됩니다. 

### 9. PCB_live_result.py
실시간으로 찍힌 PCB에 대한 추론 결과를 보여주는 화면. 기본적인 구성은 PCB_result와 동일.

<img width="1085" height="720" alt="슬라이드10" src="https://github.com/user-attachments/assets/2b9797be-a6a2-477f-a330-73bd7cd436d0" />

- map_label: PCB_result.py와 달리 Normal/Abnormal로 정상/이상 여부만 표시됩니다. (실시간 테스트 시엔 라벨이 없기 때문에, 정확한 mAP 값을 측정이 불가능하기 때문) 

## 설치 방법

이 프로젝트를 실행하기 위해 필요한 라이브러리는 다음과 같다. 

- Python 3.8+
- PyQt5
- PyTorch
- scikit-learn
- OpenCV
- matplotlib
- numpy

이 라이브러리는 아래 명령어를 사용하여 한번에 설치할 수 있다. 

```bash
pip install -r requirements.txt
```

라이브러리 설치가 완료되면 아래 단계를 통해 어플리케이션을 실행할 수 있다. 
