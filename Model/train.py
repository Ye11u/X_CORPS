import os
import shutil
import torch
import zipfile
import cv2
import random
import yaml
import numpy as np 
from pathlib import Path
from ultralytics import YOLO
try:
    from ultralytics import YOLO
    print("ultralytics 라이브러리 확인됨")
except ImportError:
    print("ultralytics 설치 필요: pip install ultralytics")

def set_seed(seed):
    """시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_yaml_config(dataset_dir):
    """yolo 학습에 필요한 .yaml 파일을 동적으로 생성"""
    print("=== YAML 설정 파일 생성 ===")
    
    yaml_content = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val', # Validation 셋이 없으면 Test 셋으로 대체하여 학습 추이 확인
        'test': 'images/test',
        'names': { # 클래스 ID와 이름을 매핑핑
            0: 'silk',
            1: 'short', 
            2: 'pad_open',
            3: 'open'
        }
    }
    
    yaml_path = Path(dataset_dir) / "pcb6.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"YAML 파일 생성: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
    return yaml_path

def validate_labels(dataset_dir):
    """라벨 파일 검증ㅣ # 학습 시작 전, 라벨 파일(.txt)에 오류가 없는지 미리 확인"""
    print("=== 라벨 파일 검증 ===")
    
    STD_ROOT = Path(dataset_dir)
    label_dirs = [STD_ROOT/"labels/train", STD_ROOT/"labels/test"]  # val 제거
    bad = []
    ids = set()
    
    for d in label_dirs:
        if not d.exists():
            continue
        for p in d.glob("*.txt"):
            with open(p, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    try:
                        cid = int(float(parts[0]))
                    except:
                        bad.append((str(p), line))
                        continue
                    ids.add(cid)
                    if cid not in {0, 1, 2, 3}: # 정의된 클래스 ID 이외의 값이 있는지 체크 
                        bad.append((str(p), line))
    
    print("발견된 클래스 ID 집합:", sorted(list(ids)))
    print("유효 범위(0~3) 외 라벨 수:", len(bad))
    if bad[:10]:
        print("예시(최대 10개):")
        for x in bad[:10]:
            print(x)

def train_model(data_yaml, epochs=50, img_size=640, batch=16, run_name="yolov8n", seed=42, save_root="./pcb_aug_js"):
    """YOLOv8n 모델을 로드하여 학습 수행"""
    print("=== YOLOv8 모델 학습 시작 ===")
    
    model = YOLO("yolov8n.pt") # 사전 학습된 nano 모델을 사용용
    save_dir = Path(save_root) / "model" / "weight"
    save_dir.mkdir(parents=True, exist_ok=True)

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(Path(save_root) / "model"),   # 결좌 저장 경로로
        name="weights",  
        val=False, # validation 과정 생략 (필요시 True)
        exist_ok=True,
        seed=seed,
        deterministic=True,
        # 데이터 증강 하이퍼파라미터: 실험을 통해 최적 증강을 찾음 (데이터가 변경되면 바꾸어야함)
        mosaic=0.3, # masaic 증강
        scale=0.2, # 이미지 스케일 변화
        translate=0.05, # 이미지 이동
        hsv_s=0.3, # 채도 변환
        hsv_v=0.2, # 명도 변환 
        degrees=2.0, # 회전 
        fliplr=0.3, # 좌우 반전
        box=7.5, # box loss 가중치
        cls=3, # class loss 가중치
        dfl=1.5, 
    )

    metrics = model.val( # 학습이 끝나면 validation을 수행 
        data=data_yaml,
        imgsz=img_size,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(save_dir),  
        name=f"{run_name}_val",
        plots=False,
    )
    # 학습 결과 파일(best.pt, last.pt)을 지정된 폴더로 정리
    yolov_dir = Path(save_root) / "model" / "weights" / "weights"
    src_last = yolov_dir / "last.pt"
    dst_last = Path(save_root) / "model" / "weights" / "last.pt"
    src_best = yolov_dir / "best.pt"
    dst_best = Path(save_root) / "model" / "weights" / "best.pt"

    if src_last.exists():
        shutil.copy(src_last, dst_last)
    else:
        print(f"last.pt 파일이 존재하지 않습니다: {src_last}")

    if src_best.exists():
        shutil.copy(src_best, dst_best)

    print("\n[학습 완료]")
    print(f"- best pt: {dst_best}")
    print(f"- last pt: {dst_last}")

    return str(dst_last)

# 학습된 모델로 테스트 셋을 추론하고, mAP 측정 및 결과 이미지 저장
def test_and_predict(weights_path, data_yaml, test_img_dir, img_size=640, conf=0.25, iou_nms=0.70):
    print("=== 테스트 및 예측 수행 ===")
    run_name = f"pred_test_conf{int(conf*100)}_iou{int(iou_nms*100)}"
    model = YOLO(weights_path)
    # 추론 속도 측정 
    import time
    start_time = time.time()
    # validation 모드로 바뀐 뒤, test set에 대해 메트릭 계산 
    test_metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=0 if torch.cuda.is_available() else "cpu",
        split="test",
        project="runs_ultra",
        name="yolov8n_test_only",
        plots=False
    )
    end_time = time.time()
    test_time = end_time - start_time
    print(f"[TEST mAP] mAP50-95={test_metrics .box.map:.4f} | mAP50={test_metrics.box.map50:.4f} | mAP75={test_metrics.box.map75:.4f}")
    print(f"Test 시간: {test_time:.2f}초")

    # 이미지에 바운딩 박스를 입힌 시각화 결과 저장 
    res = model.predict(
        source=test_img_dir,
        imgsz=img_size,
        conf=conf,
        iou=iou_nms,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs_ultra",
        name=run_name,
        line_width=2,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
        exist_ok=True
    )
    
    out_dir = Path("runs_ultra") / run_name
    print(f"\n[완료] 시각화 및 라벨 저장 폴더: {out_dir}")
    print(f"- 이미지: {out_dir}/*.jpg (또는 원본 확장자)")
    print(f"- 예측 라벨(txt): {out_dir}/labels/*.txt")
    
    return out_dir

def evaluate_background_detection(model_path, dataset_dir, img_size=640, conf=0.25, iou=0.7):
    """#YOLO 기본 val 기능은 정답이 없는(배경) 이미지에 대한 오검출(False Positive)을 명확하게 지표로 보여주지 않음. 따라서 직접 구현한 함수.
       # 정상 PCB를 불량으로 잘못 판단하는 비율(FPPI)을 계산하여 과검출 여부 확인"""
    print("=== 정상(배경) 이미지 평가 ===")
    ROOT = Path(dataset_dir)
    IMG_DIR = ROOT / "images" / "test"
    LBL_DIR = ROOT / "labels" / "test"
    
    model = YOLO(model_path)
    
    # 라벨 파일이 없거나 비어있으면 '정상(Background)' 이미지로 판단
    def is_background(img_path):
        txt = LBL_DIR / (img_path.stem + ".txt")
        if not txt.exists():
            return True
        return txt.read_text().strip() == ""
    
    # 테스트 셋에서 정상 이미지 리스트 추출
    bg_paths = [p for p in IMG_DIR.iterdir() 
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"} and is_background(p)]
    
    tn = fp = 0
    fp_boxes_total = 0
    
    for p in bg_paths:
        res = model.predict(
            source=str(p),
            imgsz=img_size,
            conf=conf,
            iou=iou,
            device=0 if torch.cuda.is_available() else "cpu",
            verbose=False
        )
        n_det = len(res[0].boxes)
        # 검출된 박스가 없어야 정상(TN)
        if n_det == 0:
            tn += 1
        # 박스가 검출되면 오검출(FP)
        else:
            fp += 1
            fp_boxes_total += n_det
    
    bg_total = len(bg_paths)
    # 특이도(Specificity): 실제 정상 중 정상으로 예측한 비율
    specificity = (tn / bg_total) if bg_total else 0.0
    # FPPI: 이미지 한 장당 평균적으로 발생하는 오검출 박스 수
    fppi = (fp_boxes_total / bg_total) if bg_total else 0.0
    
    print(f"[정상(배경) 평가 @ conf={conf}, iou={iou}]")
    print(f"- 정상 이미지 수: {bg_total}")
    print(f"- TN(무검출): {tn}")
    print(f"- FP(오검출 ≥1): {fp}")
    print(f"- 특이도 Specificity = TN / (TN+FP) = {specificity:.4f}")
    print(f"- 평균 오검출 박스 수 (FPPI): {fppi:.3f}")

def main():
    SEED = 42
    set_seed(SEED)
    
    import argparse
    parser = argparse.ArgumentParser(description="PCB YOLOv8")
    parser.add_argument('--dataset', type=str, default='pcb_original')
    args = parser.parse_args()

    print("YOLOv8 학습 시작")
    print("=" * 50)

    # GPU 환경 확인
    if torch.cuda.is_available():
        print("Gpu is available")
    else:
        print("Gpu is not available")
    
    std_root =  args.dataset

    # 설정 파일 생성성
    yaml_path = create_yaml_config(std_root)
    
    # 모델 학습 
    weights_path = train_model(
        data_yaml=yaml_path,
        epochs=50,
        img_size=640,
        batch=16,
        run_name="yolov8n",
        seed=SEED,
        save_root=args.dataset,
    )

    # 학습된 가중치 경로 설정 
    weights_path = Path(args.dataset) / "model" / "weights" / "last.pt"

    # 테스트 및 추론 
    test_img_dir = Path(args.dataset)/ 'images' / 'test'
    test_and_predict(
        weights_path=weights_path,
        data_yaml=yaml_path,
        test_img_dir=test_img_dir, 
        conf=0.25,
        iou_nms=0.70
    )
    
    # 정상 이미지 평가 (yolo 성능 평가 시 정상(no label)에 대한 성능측정은 불가하기 때문에 개별적으로 확인)
    evaluate_background_detection(
        model_path=weights_path,
        dataset_dir=str(std_root),
        conf=0.25,
        iou=0.7
    )

if __name__ == "__main__":

    main()