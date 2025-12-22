import time
import os
import shutil
import torch
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

# 전역 상수
ROOT = Path(__file__).resolve().parent

class PCBDetector:
    def __init__(self, weights_path=None):
        self.model = None
        self.weights_path = weights_path
        self.device = 0 if torch.cuda.is_available() else "cpu"
        
        if weights_path and os.path.exists(weights_path):
            self.load_model(weights_path)

    def load_model(self, weights_path):
        """모델을 메모리에 로드합니다 (1회 수행)"""
        t_start = time.perf_counter()
        print(f"[PCBDetector] 모델 로딩 시작: {weights_path}")
        self.model = YOLO(weights_path)
        # 웜업(Warmup) - 더미 데이터로 한 번 실행해두면 첫 추론 속도 향상
        # self.model(torch.zeros(1, 3, 640, 640).to(self.device)) 
        print(f"[PCBDetector] 모델 로딩 완료 ({time.perf_counter() - t_start:.4f}s)")
        self.weights_path = weights_path

    def create_yaml_config(self, dataset_dir, test_subdir="test"):
        """YAML 설정 파일 생성 로직"""
        val_path = None
        test_candidates = ['test', test_subdir]
        
        for candidate in test_candidates:
            candidate_path = Path(dataset_dir) / "images" / candidate
            if candidate_path.exists():
                val_path = f'images/{candidate}'
                break
        
        if val_path is None:
            val_path = f'images/{test_subdir}'
        
        yaml_content = {
            'path': str(dataset_dir),
            'train': 'images/train',
            'val': val_path,
            'test': f'images/{test_subdir}',
            'names': {0:'silk', 1:'short', 2:'pad_open', 3:'open'}
        }
        yaml_path = ROOT / "content" / "pcb6.yaml"
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        return yaml_path

    def run_inference(self, test_img_dir, conf=0.25, iou=0.70, show_mAP=False):
        """실제 추론을 수행하는 함수"""
        if not self.model:
            raise RuntimeError("모델이 로드되지 않았습니다. load_model을 먼저 호출하세요.")

        print(f"=== 추론 시작: {test_img_dir} ===")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_img_dir = Path(test_img_dir)
        
        # 데이터셋 경로 추론 (YAML 생성용)
        if test_img_dir.parent.name == "images":
            dataset_dir = test_img_dir.parent.parent
            test_subdir = test_img_dir.name
        else:
            dataset_dir = ROOT / "content" / "pcb_std"
            test_subdir = "test"

        # YAML 생성
        data_yaml = self.create_yaml_config(dataset_dir, test_subdir)

        # # mAP 계산 (옵션)
        #### cpu에서의 가속을 위해 해당 부분은 생략할 수 있습니다.(10초 소요)
        map50 = 0.0
        if show_mAP:
            try:
                # 라벨 파일이 있는지 간단 체크
                if any(test_img_dir.parent.parent.glob("labels/**/*.txt")):
                    metrics = self.model.val(
                        data=str(data_yaml), split="test", project="runs_ultra",
                        name="yolov8n_test_val", plots=False, device=self.device
                    )
                    map50 = metrics.box.map50
            except Exception as e:
                print(f"[WARNING] mAP 계산 실패: {e}")
        #### cpu에서의 가속을 위해 해당 부분까지는 생략할 수 있습니다.(10초 소요)

        # 결과 저장 폴더명
        run_name = f"pred_test_conf{int(conf*100)}_iou{int(iou*100)}_mAP{int(map50*1000):04d}_{timestamp}"
        
        # 예측 수행
        t_infer_start = time.perf_counter()
        res = self.model.predict(
            source=str(test_img_dir),
            imgsz=640,
            conf=conf,
            iou=iou,
            save=True,
            save_txt=True,
            save_conf=True,
            project="infer_results", # 프로젝트 루트 기준 저장
            name=run_name,
            line_width=2,
            device=self.device,
            verbose=False,
            exist_ok=True
        )
        print(f"[PERF] 순수 추론 시간: {time.perf_counter() - t_infer_start:.4f}s")
        
        # 결과 경로 반환 (infer_results/run_name)
        # 주의: predict의 project 경로는 실행 위치에 따라 달라질 수 있으므로 절대경로 보정 필요할 수 있음
        output_dir = Path("infer_results") / run_name
        return str(output_dir)

# 기존 실행 호환성 유지
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-dir', type=str)
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    
    weights = ROOT / "content" / "weights" / "last.pt"
    detector = PCBDetector(str(weights))
    if args.test_dir:
        detector.run_inference(args.test_dir, conf=args.conf)