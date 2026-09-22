import sys
import torch

def main():
    print("=" * 60)
    print(" 深度学习环境与 GPU 算力检测报告")
    print("=" * 60)
    print(f"Python 解释器路径: {sys.executable}")
    print(f"Python 版本: {sys.version.split()[0]}")
    print(f"PyTorch 核心版本: {torch.__version__}")
    
    cuda_available = torch.cuda.is_available()
    print(f"CUDA 是否可用: {'【可用】' if cuda_available else '【不可用】'}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
        print(f"检测到 GPU 数量: {gpu_count}")
        print(f"默认显卡型号: {gpu_name}")
        print(f"显卡总显存: {gpu_mem:.2f} GB")
        print(f"PyTorch 绑定的 CUDA 版本: {cuda_version}")
        print("-" * 60)
        
        # 简单张量运算验证显卡实际计算能力
        x = torch.randn(2000, 2000, device="cuda")
        y = torch.matmul(x, x)
        print(" 显卡张量矩阵乘法运算测试通过！GPU 加速已就绪！")
    else:
        print(" [警告] 当前未调用 GPU，请确认 PyTorch 是否为 cu 版本。")
    print("=" * 60)

if __name__ == "__main__":
    main()
