import os
import shutil
import torch
import yaml
import tempfile
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import time 

def create_yaml_config(dataset_dir):
    # YOLO 평가용 YAML 설정 파일 생성 (필수)
    print("\n" + "=" * 50)
    print("=== YAML 설정 파일 생성 ===")

    dataset_abs_path = Path(dataset_dir).resolve()
    # 클래스 정의 (학습 때와 동일하게 맞춰야 함)
    yaml_content = {
        'path': str(dataset_abs_path),
        'train': 'images/train',
        'val': 'images/test',
        'test': 'images/test',
        'names': {
            0: 'silk',
            1: 'short',
            2: 'pad_open',
            3: 'open'
        }
    }

    yaml_path = Path("./generated_pcb_config.yaml").resolve() # 필요하다면 경로를 바꿔주세요
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)

    print(f"YAML 파일 생성 완료: {yaml_path}")
    return str(yaml_path)

def print_class_metrics(metrics):
    # 결함 클래스별 TP/GT 개수 및 메트릭 출력
    names = metrics.names
    nc = len(names)
    matrix = metrics.confusion_matrix.matrix  # confusion matrix

    precisions = metrics.box.p
    recalls = metrics.box.r
    maps50 = metrics.box.ap50
    maps50_95 = metrics.box.ap

    print("\n[클래스별 성능 요약]")
    header = f"{'Class':<12}{'TP/GT':<12}{'Precision':<12}{'Recall':<12}{'mAP@50':<12}{'mAP@50-95':<12}"
    print(header)
    print("-" * len(header))

    total_tp, total_gt = 0, 0

    for i, name in enumerate(names.values()):
        # confusion matrix 기반 맞춘 개수 계산
        # 대각선 요소(i, i)가 True Positive(정답) 개수
        tp = matrix[i, i]
        # 해당 열(column)의 합이 Ground Truth(실제 정답) 전체 개수
        gt = matrix[:, i].sum()  
        total_tp += tp
        total_gt += gt

        p = precisions[i] if i < len(precisions) else 0.0
        r = recalls[i] if i < len(recalls) else 0.0
        m50 = maps50[i] if i < len(maps50) else 0.0
        m95 = maps50_95[i] if i < len(maps50_95) else 0.0

        # TP/GT 표시 형식 : "맞춘개수 / 전체개수"
        tp_gt_str = f"{int(tp)}/{int(gt)}"

        print(f"{name:<12}{tp_gt_str:<12}{p:<12.4f}{r:<12.4f}{m50:<12.4f}{m95:<12.4f}")

    # 전체 평균 계산 및 출력
    print("-" * len(header))
    avg_p = np.mean(precisions) if len(precisions) > 0 else 0.0
    avg_r = np.mean(recalls) if len(recalls) > 0 else 0.0
    avg_m50 = np.mean(maps50) if len(maps50) > 0 else 0.0
    avg_m95 = np.mean(maps50_95) if len(maps50_95) > 0 else 0.0
    print(f"{'ALL(평균)':<12}{int(total_tp)}/{int(total_gt):<9}{avg_p:<12.4f}{avg_r:<12.4f}{avg_m50:<12.4f}{avg_m95:<12.4f}")
    print("-" * len(header))

def test_and_predict(weights_path, data_yaml, test_img_dir, device, img_size=640, conf=0.25, iou_nms=0.70):
    print("\n" + "=" * 50)
    print("=== 테스트 세트 평가 ===")
    print("=" * 50)

    run_name = f"pred_test_conf{int(conf*100)}_iou{int(iou_nms*100)}"
    model = YOLO(weights_path)

    device=0 if torch.cuda.is_available() else "cpu"

    start_time = time.time()
    test_metrics = model.val( # mAP, Precision, Recall을 계산하는 함수: val
        data=data_yaml,
        imgsz=img_size,
        device=device,
        split="test",
        project="runs_ultra",
        name="val_total",
        plots=True,
        verbose=False,
        exist_ok=True
    )
    end_time = time.time()
    test_time = end_time - start_time
    print(f"\n[TEST mAP] mAP50-95={test_metrics.box.map:.4f} | mAP50={test_metrics.box.map50:.4f} | mAP75={test_metrics.box.map75:.4f}")
    print(f"Test 시간: {test_time:.2f}초")
    print_class_metrics(test_metrics)

    print("\n[테스트 세트 예측 결과 저장 중...]")
    model.predict( # 실제 바운딩 박스가 그려진 이미지를 저장하는 함수 
        source=test_img_dir,
        imgsz=img_size,
        conf=conf,
        iou=iou_nms,
        save=True,
        save_txt=True,
        save_conf=True,
        project="runs_ultra",
        name=run_name,
        device=device,
        verbose=False,
        exist_ok=True
    )
    print(f"[완료] 결과 저장 위치: runs_ultra/{run_name}\n")



def evaluate_per_format(model_path, data_yaml, main_dataset_dir, img_size=640):
    """파일명 접두사를 기준으로 단일 PCB 단위로 쪼갠 뒤, 성능 평가가"""
    print("\n" + "=" * 50)
    print("=== PCB 단위 개별 평가 ===")
    print("=" * 50)

    model = YOLO(model_path)
    with open(data_yaml, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    main_dataset_path = Path(main_dataset_dir).resolve()
    img_dir = main_dataset_path / config['test']
    lbl_dir = img_dir.parent.parent / "labels" / img_dir.name

    # 평가할 PCB 단위 패턴 (만약, 일부 PCB만 사용한다면 그에 맞게 수정 필요)
    format_prefixes = [ 
        "num1_A_front", "num1_A_back", 
        "num2_H_front", "num2_H_back",
        "num3_X_front", "num3_X_back",
        "num4_C_front", "num4_C_back",
        "num5_B_front", "num5_B_back", 
        "num6_D_front", "num6_D_back",
        "num7_F_front", "num7_F_back", 
        "num8_E_front", "num8_E_back",
        "num9_normal_front", "num9_normal_back"
    ]

    all_images = [p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}]
    print(f"총 {len(all_images)}개 이미지에서 형식별 평가를 수행합니다.")

    # 임시 디렉토리(Temp)를 사용하여 원본 데이터를 건드리지 않고 부분 데이터셋을 구성
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for prefix in format_prefixes:
            print(f"\n--- [{prefix}] 평가 중 ---")
            # 임시 폴더 구조 생성 
            tmp_img = tmp_path / "images" / "test"
            tmp_lbl = tmp_path / "labels" / "test"
            tmp_img.mkdir(parents=True, exist_ok=True)
            tmp_lbl.mkdir(parents=True, exist_ok=True)

            # 해당 접두사(prefix)를 가진 파일만 필터링해서 복사
            imgs = [p for p in all_images if p.name.startswith(prefix)]
            if not imgs:
                print(f"이미지 없음 → 건너뜀")
                continue

            for img in imgs:
                shutil.copy(img, tmp_img)
                lbl = lbl_dir / (img.stem + ".txt")
                if lbl.exists():
                    shutil.copy(lbl, tmp_lbl)

            # 임시 YAML 생성 (이 부분 데이터셋만 바라보도록)
            tmp_yaml = tmp_path / "temp.yaml"
            yaml_content = {
                'path': str(tmp_path.resolve()),
                'train': 'images/test', # test만 할 거라 train 경로는 더미로 넣음
                'val': 'images/test',
                'test': 'images/test',
                'names': config['names']
            }
            with open(tmp_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_content, f)

            try:
                metrics = model.val( # 부분 데이터셋에 대해 검증 수행
                    data=str(tmp_yaml),
                    imgsz=img_size,
                    device=0 if torch.cuda.is_available() else "cpu",
                    split="test",
                    project="runs_ultra",
                    name=f"val_{prefix}",
                    plots=True,
                    verbose=False,
                    exist_ok=True
                )
                print(f"[{prefix}] mAP50-95={metrics.box.map:.4f} | mAP50={metrics.box.map50:.4f}")
                print_class_metrics(metrics)
            except Exception as e:
                print(f"({prefix}): {e}")

            # 다음 루프를 위해 임시 폴더 비우기
            shutil.rmtree(tmp_img.parent, ignore_errors=True)
            shutil.rmtree(tmp_lbl.parent, ignore_errors=True)
            if tmp_yaml.exists():
                tmp_yaml.unlink()
def main():
    print("PCB 결함 검출 YOLOv8 평가 시작")
    print("=" * 50)
    
    TEST_EACH_PCB = False # PCB 단위로 성능을 보고싶다면 True로 설정
    DATASET_ROOT = "./pcb_original"
    MODEL_WEIGHTS = "./pcb_original/last.pt"
    
    yaml_path = create_yaml_config(DATASET_ROOT)
    
    # GPU 환경을 확인
    if torch.cuda.is_available():
        print("Gpu is available")
        device=0 
    else:
        print("Gpu is not available")
        device="cpu"

    with open(yaml_path, 'r', encoding='utf-8') as f:
        yconf = yaml.safe_load(f)
    test_img_dir = Path(yconf['path']) / yconf['test']
    
    test_and_predict( # 전체 데이터셋 평가
        weights_path=MODEL_WEIGHTS,
        data_yaml=yaml_path,
        test_img_dir=test_img_dir,
        device=device,
        conf=0.25,
        iou_nms=0.70
    )
    
    if TEST_EACH_PCB: # PCB 단위의 성능 측정
        evaluate_per_format(
            model_path=MODEL_WEIGHTS,
            data_yaml=yaml_path,
            main_dataset_dir=DATASET_ROOT,
            img_size=640
        )

    print("\n평가 완료")


if __name__ == "__main__":
    main()