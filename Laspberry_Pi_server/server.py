#######################################################################################
# 라즈베리파이 카메라 서버 >> Flask 웹 서버로 영상 스트리밍 및 이미지 캡처 제공
# 라즈베리파이의 GPS 폴더의 server.py로 이미 내장되어 있음 (없다면 복붙해서 사용)
#######################################################################################
from flask import Flask, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import time

app = Flask(__name__)
camera = Picamera2()

# 카메라 설정: 해상도를 최대로 설정 (2048x1536, RGB888 포맷)
config = camera.create_preview_configuration(main={"size": (2048, 1536), "format": "RGB888"})
camera.configure(config)
camera.start()
time.sleep(2)  # 카메라 초기화 대기 시간

def crop_rotated_rectangle(image, box):
    """회전된 사각형 영역을 크롭하여 정면으로 변환하는 함수"""
    # 박스의 너비와 높이 계산
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    # 목적지 좌표 (정면 사각형)
    dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    # 원근 변환 행렬 계산
    M = cv2.getPerspectiveTransform(box.astype("float32"), dst_pts)
    # 원근 변환 적용하여 크롭
    return cv2.warpPerspective(image, M, (width, height))

def detect_and_crop_pcb(image):
    """이미지에서 PCB를 검출하고 크롭하는 함수"""
    original = image.copy()
    # 그레이스케일 변환
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용 (노이즈 제거)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    # Canny 엣지 검출
    edges = cv2.Canny(blurred, 10, 90)
    # 모폴로지 연산 (닫힘) - 엣지 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # 외곽 윤곽선 찾기
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, None
    # 가장 큰 윤곽선 선택 (PCB로 추정)
    largest = max(contours, key=cv2.contourArea)
    # 최소 면적 사각형 계산
    rect = cv2.minAreaRect(largest)
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    # 회전된 사각형 영역 크롭
    cropped = crop_rotated_rectangle(original, box)
    return box, cropped

def process_image_orientation(image):
    """이미지 방향을 처리하는 함수 (세로가 가로보다 길면 90도 회전)"""
    if image is None: return None
    h, w = image.shape[:2]
    # 세로가 가로보다 길면 시계 방향으로 90도 회전
    if h > w: return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image

# while True를 돌면서 이미지를 계속 바이트 단위로 리턴(yield)함
def generate_frames():
    while True:
        # 프레임 캡처
        frame_img = camera.capture_array()
        frame_display = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)
        
        # PCB 검출 및 박스 그리기
        box, _ = detect_and_crop_pcb(frame_display)
        if box is not None:
            cv2.drawContours(frame_display, [box], -1, (0, 0, 255), 8)
            
        # 전송을 위해 이미지 크기 축소
        preview_small = cv2.resize(frame_display, (971, 601))
        
        # JPEG 인코딩
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
        ret, buffer = cv2.imencode('.jpg', preview_small, encode_param)
        frame = buffer.tobytes()

        # MJPEG 포맷에 맞춰서 패킷 만들기
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# 실시간 프레임 스트리밍 엔드포인트
@app.route('/video_feed')
def video_feed():
    """웹 브라우저가 접속하면 generate_frames 함수랑 연결해서 끊김 없이 영상 송출"""
    # mimetype을 multipart/x-mixed-replace로 설정해야 브라우저가 동영상으로 인식함
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# 기존의 단일 캡처 엔드포인트 (GUI 프로그램에서 한 장씩 요청할 때 사용)
@app.route('/frame')
def frame():
    """카메라에서 프레임을 캡처하여 JPEG 이미지로 반환하는 함수 (PCB 검출 박스 표시)"""
    frame_img = camera.capture_array()
    frame_display = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)  # RGB를 BGR로 변환
    # PCB 검출 (박스 좌표만 사용, 크롭은 하지 않음)
    box, _ = detect_and_crop_pcb(frame_display)
    if box is not None:
        # 검출된 PCB 영역에 빨간색 윤곽선 그리기
        cv2.drawContours(frame_display, [box], -1, (0, 0, 255), 8)
    # 프리뷰용으로 크기 조정
    preview_small = cv2.resize(frame_display, (971, 601)) 

    # JPEG 인코딩 (최고 품질)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode('.jpg', preview_small, encode_param)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# 이미지 캡처 엔드포인트
@app.route('/capture')
def capture():
    """카메라에서 프레임을 캡처하여 PCB를 검출하고 크롭한 후 JPEG 이미지로 반환하는 함수"""
    frame_img = camera.capture_array()
    frame_img = cv2.cvtColor(frame_img, cv2.COLOR_RGB2BGR)  # RGB를 BGR로 변환

    # PCB 검출 및 크롭
    _, cropped = detect_and_crop_pcb(frame_img)

    if cropped is None:
        return Response("No PCB detected", status=404)

    # 이미지 방향 처리 (세로가 가로보다 길면 회전)
    final_image = process_image_orientation(cropped)

    # JPEG 인코딩 (최고 품질)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    _, buffer = cv2.imencode('.jpg', final_image, encode_param)
    
    return Response(buffer.tobytes(), mimetype='image/jpeg')

# 메인 페이지 엔드포인트
@app.route('/')
def index():
    """서버 상태 확인 및 실시간 프레임을 표시하는 HTML 페이지"""
    return '''
    <html>
        <head>
            <title>Raspberry Pi PCB Detector</title>
            <meta http-equiv="refresh" content="600">  <!-- 10분마다 자동 새로고침 -->
        </head>
        <body style="background-color: black; color: white; text-align: center;">
        <h2>Server is Running (Live Stream)</h2>
            <img src="/video_feed" width="971" height="601" style="border:2px solid #444;">
        </body>
    </html>
    '''

# 서버 실행
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)  # 모든 네트워크 인터페이스에서 접근 가능, 포트 5000
    # 로그에 접속할 수 있는 url 경로가 뜸, 이 경로를 Gui 코드의 서버 url 설정 부분에 넣어줘야함 (train_test_select 화면 & live_streaming 화면 url 인자자)