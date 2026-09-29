# Screen_Detection 目标检测工程

本项目克隆自深度学习检测通用模板，已配置好 YOLOv8 与 RT-DETR 双架构支持。

## 快速上手

1. **环境准备**：
   本工程已自动通过 `.vscode/settings.json` 关联主项目的 GPU PyTorch 虚拟环境，无需重新安装庞大的依赖库！

2. **放入屏幕数据集**：
   将标注好的图片和标签分别放到：
   - 训练集：`datasets/screen_dataset/train/images/` 与 `labels/`
   - 验证集：`datasets/screen_dataset/valid/images/` 与 `labels/`
   - 检查并修改 `datasets/screen_dataset/data.yaml` 中的类别名称 (`names`)

3. **启动 YOLO 训练**：
   ```powershell
   & "C:\Users\cyz13\Documents\01_Projects\Detection_Methods_Based_on_Deep_Learning\.venv\Scripts\python.exe" -c "from scripts.train_yolo import train_yolo; train_yolo(data_yaml='datasets/screen_dataset/data.yaml', exp_name='screen_yolov8_exp', epochs=50)"
   ```

4. **启动 RT-DETR (Transformer) 训练**：
   ```powershell
   & ".\.venv\Scripts\python.exe" scripts/train_rtdetr.py
   ```

5. **启动 YOLOv8n-Hybrid 创新融合模型训练 (DySample + BiFPN + SPPF_CBAM + WIoU v3)**：
   ```powershell
   & ".\.venv\Scripts\python.exe" scripts/train_hybrid.py
   ```

