import os
import sys, time, json
import shutil
import torch
import zipfile
import cv2
import random
import yaml
from pathlib import Path
from ultralytics import YOLO

try:
    sys.stdout.reconfigure(line_buffering=True)  # 줄 단위 즉시 flush
except Exception:
    pass

def _emit(tag: str, payload: dict):
    print(f"@@{tag} {json.dumps(payload, ensure_ascii=False)}", flush=True)

# 안전 변환: loss_items (Tensor/ndarray/list/스칼라) -> list[float]
def _loss_items_to_list(obj):
    if obj is None:
        return []
    # torch.Tensor 인 경우
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().flatten().cpu().tolist()
    except Exception:
        pass
    # numpy.ndarray 인 경우
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.reshape(-1).tolist()
    except Exception:
        pass
    # 그 외 반복가능 객체
    try:
        return list(obj)
    except TypeError:
        # 스칼라
        try:
            return [float(obj)]
        except Exception:
            return []
        
# 다양한 형태의 metrics(dict/객체/내부 box객체)를 dict로 평탄화
def _dictify_metrics(obj):
    """Ultralytics validator.metrics를 플랫 dict로 변환"""
    d = {}
    if obj is None:
        return d
    try:
        box = getattr(obj, "box", None)
        if box is not None:
            # mAP
            d["mAP50-95"] = float(getattr(box, "map", 0) or 0)
            d["mAP50"]    = float(getattr(box, "map50", 0) or 0)
            # mean precision/recall (Ultralytics는 mp/mr로 들어옴)
            if hasattr(box, "mp"):
                d["precision"] = float(getattr(box, "mp", 0) or 0)
            if hasattr(box, "mr"):
                d["recall"] = float(getattr(box, "mr", 0) or 0)

        # 일부 버전은 metrics에 val 손실이 들어오기도 함
        for k in ("box_loss", "cls_loss", "dfl_loss"):
            if hasattr(obj, k):
                try:
                    d[f"val/{k}"] = float(getattr(obj, k))
                except Exception:
                    pass
    except Exception:
        pass

    # dict 형태면 병합
    if isinstance(obj, dict):
        d.update(obj)

    # 대체 키 보정
    if "mAP50" not in d:
        d["mAP50"] = float(d.get("map50", d.get("ap50", 0)) or 0)
    if "mAP50-95" not in d:
        d["mAP50-95"] = float(d.get("map", d.get("ap", 0)) or 0)

    return d





def check_gpu():
    """GPU 사용 가능 여부 확인"""
    print("=== GPU 확인 ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
    else:
        print("GPU가 없으면 학습이 매우 느립니다.")

def augment_and_copy(
    src_root, 
    dst_root, 
    aug_per_image=3, 
    offset_range=(5, 10),
    padding=20, 
    out_size=640
    ):
    
    for p in ["images/train","labels/train","images/val","labels/val","images/test","labels/test"]:
        (Path(dst_root)/p).mkdir(parents=True, exist_ok=True)

    # 카운터 변수
    orig_train_count, aug_train_count = 0, 0
    val_count, test_count = 0, 0

    def process_split(split, augment=False):
        nonlocal orig_train_count, aug_train_count, val_count, test_count

        img_dir = Path(src_root)/"images"/split
        lbl_dir = Path(src_root)/"labels"/split

        for img_file in img_dir.glob("*.jpg"):
            lbl_file = lbl_dir/f"{img_file.stem}.txt"

            # 1) 원본 복사
            shutil.copy2(img_file, Path(dst_root)/"images"/split/img_file.name)
            dst_lbl = Path(dst_root)/"labels"/split/f"{img_file.stem}.txt"
            if lbl_file.exists():
                shutil.copy2(lbl_file, dst_lbl)
            else:
                open(dst_lbl, "w").close()

            # 카운트
            if split == "train":
                orig_train_count += 1
            elif split == "val":
                val_count += 1
            elif split == "test":
                test_count += 1

            if not augment or not lbl_file.exists():
                continue

            # 2) 증강 (train만)
            img = cv2.imread(str(img_file))
            H, W = img.shape[:2]
            with open(lbl_file) as f:
                lines = [ln.strip() for ln in f if ln.strip()]

            for li, line in enumerate(lines, start=1):
                cls, x_c, y_c, bw, bh = map(float, line.split())
                x_c_px, y_c_px = x_c*W, y_c*H
                bw_px,  bh_px  = bw*W, bh*H

                for aug_idx in range(aug_per_image):
                    dx = random.randint(*offset_range) * random.choice([-1,1])
                    dy = random.randint(*offset_range) * random.choice([-1,1])
                    xc_new, yc_new = x_c_px+dx, y_c_px+dy

                    # crop 영역 (padding 포함)
                    xmin = int(xc_new - bw_px/2 - padding)
                    ymin = int(yc_new - bh_px/2 - padding)
                    xmax = int(xc_new + bw_px/2 + padding)
                    ymax = int(yc_new + bh_px/2 + padding)

                    orig_xmin, orig_ymin, orig_xmax, orig_ymax = xmin, ymin, xmax, ymax

                    # === 클리핑 처리 ===
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(W, xmax)
                    ymax = min(H, ymax)

                    # === 클리핑 발생 여부 확인 ===
                    if (xmin != orig_xmin) or (ymin != orig_ymin) or (xmax != orig_xmax) or (ymax != orig_ymax):
                        print(f"[클리핑 발생] {img_file.name} | bbox#{li} | aug{aug_idx} "
                              f"원래=({orig_xmin},{orig_ymin},{orig_xmax},{orig_ymax}) "
                              f"→ 클리핑=({xmin},{ymin},{xmax},{ymax})")

                    crop = img[ymin:ymax, xmin:xmax]
                    if crop.size == 0:
                        print(f"[경고] {img_file.name} | bbox#{li} | aug{aug_idx} crop 비어있음")
                        continue

                    crop_resized = cv2.resize(crop, (out_size, out_size))

                    new_w, new_h = xmax-xmin, ymax-ymin
                    x_c_new = (xc_new - xmin) / new_w
                    y_c_new = (yc_new - ymin) / new_h
                    bw_new  = bw_px / new_w
                    bh_new  = bh_px / new_h

                    out_img = f"{img_file.stem}_b{li}_aug{aug_idx}.jpg"
                    out_lbl = f"{img_file.stem}_b{li}_aug{aug_idx}.txt"

                    cv2.imwrite(str(Path(dst_root)/"images"/split/out_img), crop_resized)
                    with open(Path(dst_root)/"labels"/split/out_lbl, "w") as f:
                        f.write(f"{int(cls)} {x_c_new:.6f} {y_c_new:.6f} {bw_new:.6f} {bh_new:.6f}\n")

                    aug_train_count += 1

    print("Train 증강+복사"); process_split("train", augment=True)
    print("Val 복사");        process_split("val",   augment=False)
    print("Test 복사");       process_split("test",  augment=False)
    print("완료!")

    # 최종 결과 출력
    print("\n=== 요약 ===")
    print(f"Train 원본 이미지 수 : {orig_train_count}")
    print(f"Train 증강 이미지 수 : {aug_train_count}")
    print(f"Train 총합           : {orig_train_count + aug_train_count}")
    print(f"Val 총합             : {val_count}")
    print(f"Test 총합            : {test_count}")

def extract_datasets(original_zip="pcb_data.zip", augmented_zip="pcb_data2.zip", 
                    data_root="./content/pcb_raw"):
    """두 데이터셋 압축 해제 (원본 + 증강)"""
    print(f"=== 데이터셋 압축 해제 ===")
    
    # 디렉토리 생성
    original_root = os.path.join(data_root, "original")
    augmented_root = os.path.join(data_root, "augmented")
    
    os.makedirs(original_root, exist_ok=True)
    os.makedirs(augmented_root, exist_ok=True)
    
    # 원본 데이터셋 압축 해제
    print(f"원본 데이터셋 해제: {original_zip}")
    with zipfile.ZipFile(original_zip, 'r') as zip_ref:
        zip_ref.extractall(original_root)
    
    # 증강 데이터셋 압축 해제
    print(f"증강 데이터셋 해제: {augmented_zip}")
    with zipfile.ZipFile(augmented_zip, 'r') as zip_ref:
        zip_ref.extractall(augmented_root)
    
    print(f"압축 해제 완료:")
    print(f"- 원본: {original_root}")
    print(f"- 증강: {augmented_root}")
    
    return original_root, augmented_root

def setup_dataset_structure(original_root, augmented_root, test_ratio, val_ratio):
    """데이터셋 구조 설정 및 정리 (원본 우선 train 할당)"""
    print("=== 데이터셋 구조 설정 ===")
    
    random.seed(42)
    
    def find_data_root(base_path):
        """images/labels 폴더를 찾는 함수"""
        BASE = Path(base_path)
        if (BASE/"images").exists() and (BASE/"labels").exists():
            return BASE
        
        candidates = [p for p in BASE.glob("*") 
                     if p.is_dir() and (p/"images").exists() and (p/"labels").exists()]
        if len(candidates) == 1:
            return candidates[0]
        else:
            raise AssertionError(f"images/, labels/ 폴더가 필요합니다. 후보: {[str(p) for p in candidates]}")
    
    ORIGINAL_DATA_ROOT = find_data_root(original_root)
    AUGMENTED_DATA_ROOT = find_data_root(augmented_root)
    
    print(f"원본 DATA_ROOT: {ORIGINAL_DATA_ROOT}")
    print(f"증강 DATA_ROOT: {AUGMENTED_DATA_ROOT}")
    
    CLASSES = ["good", "pad_open", "open", "short", "silk", "both"]
    
    def resolve_label_dir_for_class(data_root, cls: str):
        """클래스별 라벨 디렉토리 자동 매핑"""
        labels_dir = data_root / "labels"
        cands = []
        for d in labels_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name == cls or d.name.startswith(cls):
                cands.append(d)
        
        if any(d.name == cls for d in cands):
            return [d for d in cands if d.name == cls][0]
        return cands[0] if cands else None
    
    # 표준 구조 준비
    STD_ROOT = Path("./content/pcb_std")
    for p in ["images/all", "labels/all", "images/train", "labels/train",  "images/val", "labels/val", 
              "images/test", "labels/test"]:
        (STD_ROOT / p).mkdir(parents=True, exist_ok=True)
    
    def is_img(p: Path):
        return p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
    
    def collect_images_from_dataset(data_root, prefix=""):
        """특정 데이터셋에서 이미지/라벨 수집"""
        collected = []
        label_dir_map = {cls: resolve_label_dir_for_class(data_root, cls) for cls in CLASSES}
        images_dir = data_root / "images"
        
        for cls in CLASSES:
            img_dir = images_dir / cls
            if not img_dir.exists():
                print(f"[경고] 이미지 폴더 없음: {img_dir} (건너뜀)")
                continue
            
            lbl_dir = label_dir_map.get(cls)
            for img_path in sorted(img_dir.glob("*")):
                if not is_img(img_path):
                    continue
                
                base = img_path.stem
                # 이름 충돌 방지를 위해 prefix 추가
                new_name = f"{prefix}{img_path.name}" if prefix else img_path.name
                new_base = f"{prefix}{base}" if prefix else base
                
                # 라벨 파일 경로
                cand = (lbl_dir / f"{base}.txt") if lbl_dir else None
                
                collected.append({
                    'img_path': img_path,
                    'img_name': new_name,
                    'lbl_name': f"{new_base}.txt",
                    'lbl_path': cand,
                    'class': cls
                })
        
        return collected
    
    # 원본 및 증강 데이터 수집
    print("원본 데이터 수집 중...")
    original_data = collect_images_from_dataset(ORIGINAL_DATA_ROOT, prefix="")
    
    print("증강 데이터 수집 중...")
    augmented_data = collect_images_from_dataset(AUGMENTED_DATA_ROOT, prefix="aug_")
    
    print(f"원본 이미지 수: {len(original_data)}")
    print(f"증강 이미지 수: {len(augmented_data)}")
    
    all_data = original_data + augmented_data
    total = 0
    
    for item in all_data:
        # 이미지 복사
        dst_img = STD_ROOT / "images" / "all" / item['img_name']
        shutil.copy2(item['img_path'], dst_img)
        
        # 라벨 복사 또는 빈 파일 생성
        dst_lbl = STD_ROOT / "labels" / "all" / item['lbl_name']
        if item['lbl_path'] and item['lbl_path'].exists():
            shutil.copy2(item['lbl_path'], dst_lbl)
        else:
            open(dst_lbl, "w").close()
        
        total += 1
    
    print(f"총 수집 이미지 수: {total}")
    
    random.shuffle(augmented_data)  # 증강 데이터만 섞기

    train_data = original_data.copy()
    remaining_data = augmented_data.copy()

    total_images = len(original_data) + len(augmented_data)
    target_train_size = int(total_images * (1 - val_ratio - test_ratio))
    need_more = target_train_size - len(original_data)
    target_val_size = int(total_images * val_ratio)

    if need_more > 0 and need_more <= len(augmented_data):
        additional_train = remaining_data[:need_more]
        train_data.extend(additional_train)
        val_data = remaining_data[need_more: need_more + target_val_size]
        test_data = remaining_data[need_more + target_val_size: ]
    else:
        train_data.extend(remaining_data)
        test_data = []
        val_data = []

    # 데이터 분할 결과 출력
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data
    }
    
    print(f"분할 결과:")
    print(f"- train: {len(train_data)} (원본: {len([x for x in train_data if not x['img_name'].startswith('aug_')])}, 증강: {len([x for x in train_data if x['img_name'].startswith('aug_')])})")
    print(f"- test: {len(test_data)} (원본: {len([x for x in test_data if not x['img_name'].startswith('aug_')])}, 증강: {len([x for x in test_data if x['img_name'].startswith('aug_')])})")
    print(f"- val: {len(val_data)} (원본: {len([x for x in val_data if not x['img_name'].startswith('aug_')])}, 증강: {len([x for x in val_data if x['img_name'].startswith('aug_')])})")
    
    # 실제 파일 복사
    for split_name, data_list in splits.items():
        for item in data_list:
            # 이미지 복사
            src_img = STD_ROOT / "images" / "all" / item['img_name']
            dst_img = STD_ROOT / f"images/{split_name}" / item['img_name']
            shutil.copy2(src_img, dst_img)
            
            # 라벨 복사
            src_lbl = STD_ROOT / "labels" / "all" / item['lbl_name']
            dst_lbl = STD_ROOT / f"labels/{split_name}" / item['lbl_name']
            shutil.copy2(src_lbl, dst_lbl)
    
    return STD_ROOT

def create_yaml_config(dataset_dir):
    """YOLO 학습용 YAML 설정 파일 생성"""
    print("=== YAML 설정 파일 생성 ===")
    
    yaml_content = {
        'path': str(dataset_dir),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'names': {
            0: 'silk',
            1: 'short', 
            2: 'pad_open',
            3: 'open'
        }
    }
    
    yaml_path = "./content/pcb6.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"YAML 파일 생성: {yaml_path}")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        print(f.read())
    
    return yaml_path

def validate_labels(dataset_dir):
    """라벨 파일 검증"""
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
                    if cid not in {0, 1, 2, 3}:
                        bad.append((str(p), line))
    
    print("발견된 클래스 ID 집합:", sorted(list(ids)))
    print("유효 범위(0~3) 외 라벨 수:", len(bad))
    if bad[:10]:
        print("예시(최대 10개):")
        for x in bad[:10]:
            print(x)

def train_model(data_yaml, epochs=50, img_size=640, batch=16, run_name="yolov8n"):
    print("=== YOLOv8 모델 학습 시작 ===")
    model = YOLO("yolov8n.pt")

    def _on_train_batch_end(trainer):
        li = _loss_items_to_list(getattr(trainer, "loss_items", None))
        bl = float(li[0]) if len(li) > 0 else None
        cl = float(li[1]) if len(li) > 1 else None
        dl = float(li[2]) if len(li) > 2 else None
        e = int(getattr(trainer, "epoch", 0))
        i = int(getattr(trainer, "batch_i", 0))
        n = int(getattr(trainer, "nb", 0)) or 0
        if n <= 0:
            try:
                n = len(getattr(trainer, "train_loader", []))
            except Exception:
                n = 1
        _emit("BATCH", {
            "epoch": e, "i": i, "n": n, "epochs": int(epochs),
            "losses": {
                "train/box_loss": bl,
                "train/cls_loss": cl,
                "train/dfl_loss": dl,
                "total_loss": (bl or 0) + (cl or 0) + (dl or 0),
            }
        })

    def _on_fit_epoch_end(trainer):
        cur = int(getattr(trainer, "epoch", 0)) + 1
        _emit("EPOCH", {"cur": cur, "tot": int(epochs)})

    def _on_val_end(trainer):
        cur_epoch = int(getattr(trainer, "epoch", 0)) + 1
        m = _dictify_metrics(getattr(getattr(trainer, "validator", None), "metrics", None))
        if not m:
            m = _dictify_metrics(getattr(trainer, "metrics", None))
        _emit("ACC", {
            "model_type": "yolov8",
            "epoch": cur_epoch,
            "precision": float(m.get("precision", 0) or 0),
            "recall":    float(m.get("recall", 0) or 0),
            "mAP50":     float(m.get("mAP50", 0) or 0),
            "mAP50-95":  float(m.get("mAP50-95", 0) or 0),
            # val 손실은 있으면 보내고, 없으면 누락(None) 처리됨
            "val/box_loss": m.get("val/box_loss", None),
            "val/cls_loss": m.get("val/cls_loss", None),
            "val/dfl_loss": m.get("val/dfl_loss", None),
        })

    # train_model() 안
    def _on_train_epoch_end(trainer):
        # 에폭 평균 손실(ultralytics가 계산해 둔 값을 활용 가능 / 없으면 마지막 배치 값이라도 송출)
        try:
            li = _loss_items_to_list(getattr(trainer, "loss_items", None))
            bl = float(li[0]) if len(li) > 0 else None
            cl = float(li[1]) if len(li) > 1 else None
            dl = float(li[2]) if len(li) > 2 else None
        except Exception:
            bl = cl = dl = None

        cur_epoch = int(getattr(trainer, "epoch", 0)) + 1
        _emit("EPOCH_LOSS", {"epoch": cur_epoch,
                            "train/box_loss": bl,
                            "train/cls_loss": cl,
                            "train/dfl_loss": dl})

    model.add_callback("on_train_epoch_end", _on_train_epoch_end)


    model.add_callback("on_train_batch_end", _on_train_batch_end)
    model.add_callback("on_fit_epoch_end",  _on_fit_epoch_end)
    model.add_callback("on_val_end",        _on_val_end)

    # 프로젝트(저장) 폴더를 절대경로로 명시
    project_dir = Path(os.getcwd()) / "runs_ultra"
    project_dir.mkdir(parents=True, exist_ok=True)

    # 매번 새로운 폴더로 저장 (원하시면 아래 한 줄을 run_name 그대로 사용해도 OK)
    tagged_run_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"

    # UI가 바로 물어볼 수 있도록 "이번 런 폴더"를 먼저 알림
    print(f'@@RUN_DIR {json.dumps({"run_dir": str(project_dir / tagged_run_name), "results_csv": str(project_dir / tagged_run_name / "results.csv")}, ensure_ascii=False)}', flush=True)


    # === 학습 ===
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(project_dir),
        name=tagged_run_name,
        exist_ok=True,
        mosaic=0.3, scale=0.2, translate=0.05, hsv_s=0.3, hsv_v=0.2, degrees=2.0, fliplr=0.3,
        box=7.5, cls=0.75, dfl=1.5,
    )

    project_dir = Path(os.getcwd()) / "runs_ultra"     # 이미 위에 있음
    save_dir = project_dir / tagged_run_name           # ★ 여기!
    best_pt  = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    # === (중요) 검증은 "새 YOLO 인스턴스"로, 별도 폴더에 저장 → results.csv 덮어쓰지 않게
    val_model = YOLO(str(best_pt if best_pt.exists() else (last_pt if last_pt.exists() else "yolov8n.pt")))
    _ = val_model.val(
        data=data_yaml,
        imgsz=img_size,
        device=0 if torch.cuda.is_available() else "cpu",
        project="runs_ultra",
        name=f"{run_name}_val",
        plots=False,
        exist_ok=True,
    )

    print("\n[학습 완료]")
    print(f"- best pt: {best_pt}")
    print(f"- last pt: {last_pt}")
    return str(best_pt if best_pt.exists() else last_pt)




def test_and_predict(weights_path, data_yaml, test_img_dir, img_size=640, conf=0.25, iou_nms=0.70):
    """테스트 및 예측 수행"""
    print("=== 테스트 및 예측 수행 ===")
    
    run_name = f"pred_test_conf{int(conf*100)}_iou{int(iou_nms*100)}"
    
    # 모델 로드
    model = YOLO(weights_path)
    
    # 테스트 mAP 재확인
    test_metrics = model.val(
        data=data_yaml,
        imgsz=img_size,
        device=0 if torch.cuda.is_available() else "cpu",
        split="test",
        project="runs_ultra",
        name="yolov8n_test_only",
        plots=False
    )
    print(f"[TEST mAP] mAP50-95={test_metrics .box.map:.4f} | mAP50={test_metrics.box.map50:.4f} | mAP75={test_metrics.box.map75:.4f}")
    
    # 테스트 예측 & 저장
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
    """정상(배경) 이미지에 대한 TN/FP 계산"""
    print("=== 정상(배경) 이미지 평가 ===")
    
    ROOT = Path(dataset_dir)
    IMG_DIR = ROOT / "images" / "test"
    LBL_DIR = ROOT / "labels" / "test"
    
    model = YOLO(model_path)
    
    def is_background(img_path):
        """이미지가 정상(배경)인지 판단"""
        txt = LBL_DIR / (img_path.stem + ".txt")
        if not txt.exists():
            return True
        return txt.read_text().strip() == ""
    
    # 정상 이미지 찾기
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
        if n_det == 0:
            tn += 1
        else:
            fp += 1
            fp_boxes_total += n_det
    
    bg_total = len(bg_paths)
    specificity = (tn / bg_total) if bg_total else 0.0
    fppi = (fp_boxes_total / bg_total) if bg_total else 0.0
    
    print(f"[정상(배경) 평가 @ conf={conf}, iou={iou}]")
    print(f"- 정상 이미지 수: {bg_total}")
    print(f"- TN(무검출): {tn}")
    print(f"- FP(오검출 ≥1): {fp}")
    print(f"- 특이도 Specificity = TN / (TN+FP) = {specificity:.4f}")
    print(f"- 평균 오검출 박스 수 (FPPI): {fppi:.3f}")


def main():
    print("PCB 결함 검출 YOLOv8 학습 시작")
    print("=" * 50)
    check_gpu()
    try:
        from ultralytics import YOLO
        print("ultralytics 라이브러리 확인됨")
    except ImportError:
        print("ultralytics 설치 필요: pip install ultralytics")
        return

    # =========================
    # ★ 데모 모드(고정 데이터셋)
    # =========================
    std_root = Path("./content/pcb_std")   # ← 표준 데이터셋 고정
    run_name = "yolov8n"                   # ← runs_ultra/yolov8n/results.csv (모니터링과 일치)

    print("[DEMO MODE] Using standard dataset only:", std_root.resolve())
    print("[DEMO MODE] Ignoring uploaded zips & run_config.json")

    # YAML 생성/라벨 검증/학습/평가
    yaml_path = create_yaml_config(std_root)
    validate_labels(std_root)

    # 에폭 원하는 값으로 (모니터링 그래프용이면 30~50 추천)
    weights_path = train_model(
        data_yaml=yaml_path,
        epochs=50,
        img_size=640,
        batch=32,
        run_name=run_name
    )

    test_img_dir = std_root / "images" / "test"
    test_and_predict(weights_path, yaml_path, str(test_img_dir), conf=0.25, iou_nms=0.70)
    evaluate_background_detection(weights_path, std_root, conf=0.25, iou=0.7)
    print("\n모든 작업이 완료되었습니다!")

if __name__ == "__main__":
    main()