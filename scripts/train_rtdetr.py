import os
from pathlib import Path
from ultralytics import RTDETR

BASE_DIR = Path(__file__).resolve().parent.parent

def train_rtdetr(
    data_yaml=None,
    model_weight="rtdetr-l.pt",
    epochs=50,
    batch_size=8,
    imgsz=640,
    device=0,
    exp_name="rtdetr_screen_experiment"
):
    """
    训练 RT-DETR 屏幕缺陷目标检测模型
    :param data_yaml: 数据集配置文件路径 (data.yaml)
    :param model_weight: 预训练权重名称 (如 rtdetr-l.pt)
    :param epochs: 训练轮数
    :param batch_size: 批次大小 (RTX 4070 建议 8)
    :param imgsz: 输入图像分辨率
    :param device: 显卡编号 (0 为独立显卡)
    :param exp_name: 实验输出名称
    """
    if data_yaml is None:
        data_yaml = str(BASE_DIR / "datasets" / "screen_dataset" / "data.yaml")

    print("=" * 60)
    print(f" 开始训练 Transformer 屏幕缺陷模型 (RT-DETR): {model_weight}")
    print(f" 实验名称: {exp_name}")
    print(f" 数据集配置: {data_yaml}")
    print(f" 训练轮数: {epochs} | 批次: {batch_size} | 图像尺寸: {imgsz}")
    print("=" * 60)

    weight_in_dir = BASE_DIR / "weights" / Path(model_weight).name
    target_weight = str(weight_in_dir) if weight_in_dir.exists() else model_weight

    model = RTDETR(target_weight)

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=device,
        workers=4,
        optimizer="AdamW",
        lr0=0.0001,
        project=str(BASE_DIR / "runs" / "detect"),
        name=exp_name,
        exist_ok=True,
        plots=True,
        save=True
    )
    
    print("\n" + "=" * 60)
    print(" RT-DETR (Transformer) 屏幕缺陷模型训练完成！")
    print(f" 最佳权重保存在: {BASE_DIR}/runs/detect/{exp_name}/weights/best.pt")
    print("=" * 60)
    return results

if __name__ == "__main__":
    train_rtdetr()
