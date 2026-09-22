"""
此脚本用于在没有立即下载完整大体积数据集前，
自动生成一个迷你的测试数据集目录结构与少量样本图像，
用于一键跑通 train_yolo.py / train_rtdetr.py 验证 GPU 训练管线无 Bug。
"""
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "datasets" / "plant_disease"

def create_sample_dataset():
    print(f"正在创建示例数据集目录: {DATASET_DIR}")
    
    # 创建目录结构
    for split in ["train", "valid"]:
        (DATASET_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
        
        # 为 train 生成 6 张虚拟叶片图片，为 valid 生成 2 张
        num_samples = 6 if split == "train" else 2
        for i in range(num_samples):
            # 创建绿色底色的“叶片”图像 (640x640)
            img = Image.new("RGB", (640, 640), color=(34, 139, 34))
            draw = ImageDraw.Draw(img)
            
            # 画一个褐色圆形斑点代表“病斑” (早疫病)
            cx, cy, r = 200 + i * 30, 200 + i * 20, 40
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(139, 69, 19), outline=(0, 0, 0))
            
            img_path = DATASET_DIR / split / "images" / f"sample_{i:02d}.jpg"
            img.save(img_path)
            
            # 写入对应的 YOLO 格式标注: class_id x_center y_center width height (归一化 0~1)
            norm_cx, norm_cy = cx / 640.0, cy / 640.0
            norm_w, norm_h = (2 * r) / 640.0, (2 * r) / 640.0
            
            label_path = DATASET_DIR / split / "labels" / f"sample_{i:02d}.txt"
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")
                
    # 写入 data.yaml
    yaml_content = f"""# 示例数据集配置文件
path: {DATASET_DIR.as_posix()}
train: train/images
val: valid/images

nc: 3
names:
  0: 'Early_Blight'
  1: 'Late_Blight'
  2: 'Healthy'
"""
    yaml_path = DATASET_DIR / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
        
    print(" 示例数据集与 data.yaml 创建成功！可立即用于测试训练！")

if __name__ == "__main__":
    create_sample_dataset()
