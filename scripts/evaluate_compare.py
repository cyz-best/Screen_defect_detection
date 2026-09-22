import os
import time
from pathlib import Path
import torch
import pandas as pd
from ultralytics import YOLO, RTDETR

BASE_DIR = Path(__file__).resolve().parent.parent

def benchmark_single_model(model_cls, weights_path, model_name, data_yaml):
    print(f"\n>> 正在评测模型: {model_name} (权重: {weights_path}) ...")
    if not os.path.exists(weights_path):
        print(f" [跳过] 找不到权重文件 {weights_path}，请确认是否已完成训练。")
        return None

    model = model_cls(weights_path)

    val_results = model.val(data=data_yaml, imgsz=640, device=0, split='test', verbose=False)
    map50 = val_results.box.map50
    map50_95 = val_results.box.map
    precision = val_results.box.mp
    recall = val_results.box.mr

    dummy_input = torch.rand(1, 3, 640, 640).cuda()
    for _ in range(15):
        _ = model(dummy_input, verbose=False)
    
    torch.cuda.synchronize()
    start_time = time.time()
    iters = 100
    for _ in range(iters):
        _ = model(dummy_input, verbose=False)
    torch.cuda.synchronize()
    
    avg_latency_ms = ((time.time() - start_time) / iters) * 1000
    fps = 1000 / avg_latency_ms

    return {
        "模型名称": model_name,
        "网络架构": "CNN 局部卷积" if "YOLO" in model_name else "Transformer 全局自注意力",
        "mAP@0.5 (%)": round(map50 * 100, 2),
        "mAP@0.5:0.95 (%)": round(map50_95 * 100, 2),
        "精确率 Precision (%)": round(precision * 100, 2),
        "召回率 Recall (%)": round(recall * 100, 2),
        "平均推理延迟 (ms)": round(avg_latency_ms, 2),
        "推理帧率 FPS": round(fps, 1)
    }

def main():
    data_yaml = str(BASE_DIR / "datasets" / "screen_dataset" / "data.yaml")
    
    yolo_weight = str(BASE_DIR / "runs" / "detect" / "yolov8_screen_experiment" / "weights" / "best.pt")
    rtdetr_weight = str(BASE_DIR / "runs" / "detect" / "rtdetr_screen_experiment" / "weights" / "best.pt")

    results = []
    
    res_yolo = benchmark_single_model(YOLO, yolo_weight, "YOLOv8n (屏幕缺陷)", data_yaml)
    if res_yolo:
        results.append(res_yolo)

    res_rtdetr = benchmark_single_model(RTDETR, rtdetr_weight, "RT-DETR-L (屏幕缺陷)", data_yaml)
    if res_rtdetr:
        results.append(res_rtdetr)

    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 70)
        print(" 【实验成果】屏幕缺陷检测 YOLO vs Transformer 对比评测表格")
        print("=" * 70)
        print(df.to_markdown(index=False))
        
        output_csv = BASE_DIR / "runs" / "screen_model_comparison_report.csv"
        df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n对比报告已自动保存到: {output_csv}")
    else:
        print("\n尚未检测到训练好的模型权重，请先运行 train_yolo.py 或 train_rtdetr.py！")

if __name__ == "__main__":
    main()
