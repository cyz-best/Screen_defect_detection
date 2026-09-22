import os
import json
from pathlib import Path
import gradio as gr
from ultralytics import YOLO, RTDETR

BASE_DIR = Path(__file__).resolve().parent.parent

# 权重路径配置
YOLO_CUSTOM_WEIGHT = BASE_DIR / "runs" / "detect" / "yolov8_screen_experiment" / "weights" / "best.pt"
RTDETR_CUSTOM_WEIGHT = BASE_DIR / "runs" / "detect" / "rtdetr_screen_experiment" / "weights" / "best.pt"

# 外部屏幕缺陷工业知识库路径
KB_PATH = BASE_DIR / "knowledge_base" / "screen_defect_kb.json"

def load_knowledge_base():
    if KB_PATH.exists():
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取知识库异常: {e}")
    return {}

DEFECT_KB = load_knowledge_base()

loaded_models = {}

def get_model(model_name):
    """根据选择动态加载模型权重"""
    if model_name not in loaded_models:
        if "YOLO" in model_name:
            if YOLO_CUSTOM_WEIGHT.exists():
                path = str(YOLO_CUSTOM_WEIGHT)
            elif (BASE_DIR / "weights" / "yolov8n.pt").exists():
                path = str(BASE_DIR / "weights" / "yolov8n.pt")
            else:
                path = "yolov8n.pt"
            print(f"正在加载 YOLO 模型: {path}")
            loaded_models[model_name] = YOLO(path)
        else:
            if RTDETR_CUSTOM_WEIGHT.exists():
                path = str(RTDETR_CUSTOM_WEIGHT)
            elif (BASE_DIR / "weights" / "rtdetr-l.pt").exists():
                path = str(BASE_DIR / "weights" / "rtdetr-l.pt")
            else:
                path = "rtdetr-l.pt"
            print(f"正在加载 RT-DETR 模型: {path}")
            loaded_models[model_name] = RTDETR(path)
    return loaded_models[model_name]

def predict(image, model_choice, conf_thresh, iou_thresh):
    if image is None:
        return None, "⚠️ 请先上传待检测的手机屏幕实拍图片！"

    model = get_model(model_choice)
    results = model.predict(source=image, conf=conf_thresh, iou=iou_thresh, verbose=False)
    res = results[0]
    
    # 绘制带检测框和置信度标签的高清图像
    annotated_img = res.plot()

    # 提取检测结果与排查建议
    detected_list = []
    cards_list = []
    
    if len(res.boxes) == 0:
        diagnosis_md = """
### ✅ 检测结果：【良品 / 暂未发现明显表面缺陷】
* **质检判定**：在当前置信度阈值下，未检测到油污、划痕或严重污渍缺陷。
* **建议**：如目测有极其微弱瑕疵，可尝试将左侧“置信度阈值”适当调低（如 0.15~0.20）重新检测。
"""
    else:
        total_defects = len(res.boxes)
        cards_list.append(f"### ⚠️ 质检判定：【不良品 - 检测到 {total_defects} 处屏幕表面缺陷】\n")
        
        for idx, box in enumerate(res.boxes, start=1):
            cls_id = int(box.cls[0].item())
            cls_name = res.names[cls_id]
            conf = float(box.conf[0].item())
            
            kb_info = DEFECT_KB.get(cls_name, {})
            zh_name = kb_info.get("chinese_name", cls_name)
            def_type = kb_info.get("defect_type", "工业外观缺陷")
            severity = kb_info.get("severity", "需质检复核")
            cause = kb_info.get("cause", "制程工序异常或外界接触划碰")
            repair = kb_info.get("repair_advice", "按照无尘车间标准清洗擦拭或降级返工")
            process = kb_info.get("process_optimization", "巡检传送治具、机械手保护垫与洁净车间环境")

            card = f"""
---
#### 🔍 缺陷 #{idx}：{zh_name}  |  检测置信度：`{conf:.2%}`
* **瑕疵类型**：{def_type}  |  **严重等级**：`{severity}`
* **潜在致因**：{cause}
* **现场处置建议**：{repair}
* **产线工艺改善**：{process}
"""
            cards_list.append(card)
        
        diagnosis_md = "\n".join(cards_list)

    return annotated_img, diagnosis_md

# 准备测试样例图片列表
valid_img_dir = BASE_DIR / "datasets" / "screen_dataset" / "valid" / "images"
examples = []
if valid_img_dir.exists():
    for f in list(valid_img_dir.glob("*.jpg"))[:4]:
        examples.append([str(f)])

# 构建现代化 Gradio 界面
custom_css = """
.gradio-container { max-width: 1280px !important; margin: auto; }
#title-header { text-align: center; margin-bottom: 20px; }
#result-card { background: #fdfdfd; border-radius: 8px; padding: 15px; border: 1px solid #e2e8f0; }
"""

with gr.Blocks(title="手机屏幕缺陷智能检测与工业质检系统", css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 📱 手机屏幕缺陷智能检测与工业质检系统
        ### 基于深度学习 (YOLOv8 / RT-DETR) 的高精度实时缺陷定位与产线改善建议
        """,
        elem_id="title-header"
    )

    with gr.Row():
        with gr.Column(scale=5):
            input_image = gr.Image(type="pil", label="上传手机屏幕实拍图")
            
            with gr.Row():
                model_choice = gr.Radio(
                    choices=["YOLOv8 工业检测模型 (当前已训练完成)", "RT-DETR (Transformer 架构)"],
                    value="YOLOv8 工业检测模型 (当前已训练完成)",
                    label="选择检测算法模型"
                )
            
            with gr.Row():
                conf_slider = gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05, label="置信度阈值 (Confidence)")
                iou_slider = gr.Slider(minimum=0.1, maximum=0.9, value=0.45, step=0.05, label="NMS 重叠度阈值 (IoU)")
            
            submit_btn = gr.Button("🚀 开始屏幕缺陷智能检测与分析", variant="primary", size="lg")
            
            if examples:
                gr.Examples(
                    examples=examples,
                    inputs=input_image,
                    label="点击加载屏幕缺陷测试样例图"
                )

        with gr.Column(scale=6):
            output_image = gr.Image(type="pil", label="🎯 缺陷画框定位与标签识别图")
            output_report = gr.Markdown(label="📋 工业质检报告与工艺处置卡片", elem_id="result-card")

    submit_btn.click(
        fn=predict,
        inputs=[input_image, model_choice, conf_slider, iou_slider],
        outputs=[output_image, output_report]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True)
