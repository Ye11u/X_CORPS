import os, random, shutil
from pathlib import Path
import cv2
import random

def decide_crop_region(W, H, bbox, ratio=0.9, img_name=""):
    x1, y1, x2, y2 = bbox
    bx, by = (x1+x2)/2, (y1+y2)/2
    cx, cy = W/2, H/2
    crop_w, crop_h = int(W*ratio), int(H*ratio)

    crosses_x = (x1 < cx < x2)   # 좌+우 걸침
    crosses_y = (y1 < cy < y2)   # 상+하 걸침

    print(f"\n[이미지: {img_name}] ratio={ratio:.3f}")
    print(f"  bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), "
          f"중심=({bx:.1f},{by:.1f}), 이미지 중심=({cx:.1f},{cy:.1f})")

    # --- Case 1: 한 사분면 ---
    if not crosses_x and not crosses_y:
        if bx >= cx and by < cy:       # Q1 (우상)
            x1c, y1c = W - crop_w, 0
            print("  → Q1: 왼쪽 잘라냄 + 위쪽 유지 → 아래쪽 crop")
        elif bx < cx and by < cy:      # Q2 (좌상)
            x1c, y1c = 0, 0
            print("  → Q2: 오른쪽 잘라냄 + 위쪽 유지 → 아래쪽 crop")
        elif bx < cx and by >= cy:     # Q3 (좌하)
            x1c, y1c = 0, H - crop_h
            print("  → Q3: 오른쪽 잘라냄 + 아래쪽 유지 → 위쪽 crop")
        else:                          # Q4 (우하)
            x1c, y1c = W - crop_w, H - crop_h
            print("  → Q4: 왼쪽 잘라냄 + 아래쪽 유지 → 위쪽 crop")

    # --- Case 2: 두 사분면 (좌우 걸침) ---
    elif crosses_x and not crosses_y:
        area_left  = (min(cx, x2) - x1) * (y2 - y1)
        area_right = (x2 - max(cx, x1)) * (y2 - y1)

        if by < cy:   # Q1+Q2 (위쪽)
            y1c = 0   # 아래쪽 잘라냄
            if area_left >= area_right:
                x1c = 0
                print("  → Q1+Q2: 아래쪽 crop + 왼쪽 유지 → 오른쪽 잘라냄")
            else:
                x1c = W - crop_w
                print("  → Q1+Q2: 아래쪽 crop + 오른쪽 유지 → 왼쪽 잘라냄")
        else:         # Q3+Q4 (아래쪽)
            y1c = H - crop_h   # 위쪽 잘라냄
            if area_left >= area_right:
                x1c = 0
                print("  → Q3+Q4: 위쪽 crop + 왼쪽 유지 → 오른쪽 잘라냄")
            else:
                x1c = W - crop_w
                print("  → Q3+Q4: 위쪽 crop + 오른쪽 유지 → 왼쪽 잘라냄")

    # --- Case 2: 두 사분면 (상하 걸침) ---
    elif crosses_y and not crosses_x:
        area_top    = (x2 - x1) * (min(cy, y2) - y1)
        area_bottom = (x2 - x1) * (y2 - max(cy, y1))

        if bx >= cx:   # 오른쪽(Q1+Q4)
            x1c = W - crop_w
            side = "왼쪽"
        else:          # 왼쪽(Q2+Q3)
            x1c = 0
            side = "오른쪽"

        if area_top >= area_bottom:
            y1c = H - crop_h   # 위쪽 crop
            print(f"  → 상+하: {side} 잘라냄 + 위쪽 유지 → 아래쪽 crop")
        else:
            y1c = 0   # 아래쪽 crop
            print(f"  → 상+하: {side} 잘라냄 + 아래쪽 유지 → 위쪽 crop")

    # --- Case 3: 네 사분면 ---
    else:
        x1c = (W - crop_w)//2
        y1c = (H - crop_h)//2
        print("  → 네 사분면 모두 걸침: 중앙 균등 crop")

    x2c, y2c = x1c + crop_w, y1c + crop_h
    return (int(x1c), int(y1c), int(x2c), int(y2c))

def augment_dataset(src_root, dst_root, n_aug=2, ratio_range=(0.8,0.9), out_size=640):
    src_root, dst_root = Path(src_root), Path(dst_root)

    # 카운터
    count_train_orig_src = 0
    count_train_orig_copied = 0
    count_train_aug = 0
    count_skipped_no_label = 0
    count_skipped_aug = 0

    for split in ["train", "val", "test"]:
        img_dir = src_root/"images"/split
        lbl_dir = src_root/"labels"/split
        out_img_dir = dst_root/"images"/split
        out_lbl_dir = dst_root/"labels"/split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        for img_path in img_paths:
            name = img_path.stem
            label_path = lbl_dir / f"{name}.txt"
            if not label_path.exists():
                if split == "train":
                    count_skipped_no_label += 1
                continue

            # === val/test: 그대로 복사 ===
            if split in ["val","test"]:
                shutil.copy(img_path, out_img_dir/f"{name}.jpg")
                shutil.copy(label_path, out_lbl_dir/f"{name}.txt")
                continue

            # === train: 원본 복사 + 증강 추가 ===
            count_train_orig_src += 1
            # 이미지 읽고 resize (out_size × out_size)
            img = cv2.imread(str(img_path))
            img_resized = cv2.resize(img, (out_size, out_size))
            cv2.imwrite(str(out_img_dir/f"{name}.jpg"), img_resized)
            shutil.copy(label_path, out_lbl_dir/f"{name}.txt")
            count_train_orig_copied += 1

            img = cv2.imread(str(img_path))
            H, W = img.shape[:2]

            with open(label_path, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]

            bboxes = []
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls, cx, cy, w, h = map(float, parts)
                bx, by = cx*W, cy*H
                bw, bh = w*W, h*H
                x1, y1 = bx-bw/2, by-bh/2
                x2, y2 = bx+bw/2, by+bh/2
                bboxes.append((x1,y1,x2,y2,int(cls)))

            if not bboxes:
                continue

            # 기준 bbox: 가장 큰 것
            x1, y1, x2, y2, _ = max(bboxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

            for k in range(n_aug):
                ratio = random.uniform(*ratio_range)
                x1c, y1c, x2c, y2c = decide_crop_region(
                    W, H, (x1,y1,x2,y2), ratio, img_name=f"{split}/{name}_aug{k}"
                )

                crop = img[y1c:y2c, x1c:x2c]
                crop_resized = cv2.resize(crop, (out_size,out_size))

                crop_w, crop_h = x2c-x1c, y2c-y1c
                new_labels = []
                for (bx1,by1,bx2,by2,cls) in bboxes:
                    nx1, ny1 = bx1-x1c, by1-y1c
                    nx2, ny2 = bx2-x1c, by2-y1c
                    nx1, ny1 = max(0,nx1), max(0,ny1)
                    nx2, ny2 = min(crop_w,nx2), min(crop_h,ny2)
                    if nx2 <= nx1 or ny2 <= ny1:
                        print(f"  !!!!! [이미지: {name}_aug{k}] bbox 소멸 → 증강본 저장 스킵")
                        new_labels = []
                        break
                    sx, sy = out_size/crop_w, out_size/crop_h
                    rx1, ry1 = nx1*sx, ny1*sy
                    rx2, ry2 = nx2*sx, ny2*sy
                    rcx, rcy = (rx1+rx2)/2/out_size, (ry1+ry2)/2/out_size
                    rw, rh = (rx2-rx1)/out_size, (ry2-ry1)/out_size
                    new_labels.append(f"{cls} {rcx:.6f} {rcy:.6f} {rw:.6f} {rh:.6f}\n")

                if not new_labels:
                    count_skipped_aug += 1
                    continue

                out_img_path = out_img_dir / f"{name}_aug{k}.jpg"
                out_lbl_path = out_lbl_dir / f"{name}_aug{k}.txt"
                cv2.imwrite(str(out_img_path), crop_resized)
                with open(out_lbl_path,"w") as f:
                    f.writelines(new_labels)

                count_train_aug += 1

    # 결과 출력
    print("\n==================== 요약 ====================")
    print(f"Train (src) 원본 이미지 수 : {count_train_orig_src}")
    print(f"Train (dst) 원본 복사 수   : {count_train_orig_copied}")
    print(f"Train 증강 생성 수         : {count_train_aug}")
    print(f"Train 최종 총합 (dst)      : {count_train_orig_copied + count_train_aug}")
    print(f"스킵(라벨 없음/비어있음)   : {count_skipped_no_label}")
    print(f"스킵(bbox 소멸)            : {count_skipped_aug}")
    print("Val/Test는 증강 없이 1:1 복사 완료")
    print("==============================================")


def main():
    src_root="./final/pcb_original_with_good" #원본 데이터셋
    dst_root="./final/pcb_original_with_good_and_js_aug" #증강된 데이터셋을 저장할 경로
    
    random.seed(42) # 시드 고정

    augment_dataset( # 증강 함수 
        src_root=src_root,
        dst_root=dst_root,
        n_aug=1,
        ratio_range=(0.9,0.95),
        out_size=160
    )
    
    print('\n증강 완료')

if __name__ == "__main__":
    main()   