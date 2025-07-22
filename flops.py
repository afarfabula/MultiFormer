import torch
import torch.nn as nn
from thop import profile
import time
from models.wisppn_resnet import *
from models.CDKformer import conformermulti

def calculate_model_metrics(model, input_size):
    """
    计算模型的参数量、FLOPS、MACTS、推理速度和内存占用
    Args:
        model: 要评估的模型
        input_size: 输入张量的大小
    Returns:
        dict: 包含各项指标的字典
    """
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    
    # 计算FLOPS和MACTS
    flops, macs = profile(model, inputs=(dummy_input,), verbose=False)
    
    # 计算推理时间
    model.eval()
    with torch.no_grad():
        # 预热
        for _ in range(10):
            _ = model(dummy_input)
        
        # 正式计时
        start_time = time.time()
        for _ in range(100):
            _ = model(dummy_input)
        end_time = time.time()
    
    inference_time = (end_time - start_time) / 100
    
    # 计算内存占用
    mem_params = sum([param.nelement() * param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement() * buf.element_size() for buf in model.buffers()])
    mem_total = mem_params + mem_bufs
    
    return {
        'total_params': total_params,
        'params_size(MB)': mem_params / (1024 ** 2),
        'total_memory(MB)': mem_total / (1024 ** 2),
        'flops(GFLOPS)': flops / (10**9),
        'macs': macs,
        'fps': 1 / inference_time
    }

def compare_models():
    """比较三个模型的性能指标"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型实例
    resnet_fd = ResNetFD(ResidualBlock, [3, 4, 6,0]).to(device).to(device)
    resnet = ResNet(ResidualBlock, layers=[3, 4, 6, 3]).to(device)
    cdkformer = conformermulti(num_classes=36, ind=0).to(device)
    
    # 定义输入尺寸
    resnet_fd_input = (1, 3, 36, 36)
    resnet_input = (1, 3, 36, 36)  # 假设与ResNetFD相同
    cdkformer_input = (1, 1, 12, 51)
    
    # 计算指标
    resnet_fd_metrics = calculate_model_metrics(resnet_fd, resnet_fd_input)
    resnet_metrics = calculate_model_metrics(resnet, resnet_input)
    cdkformer_metrics = calculate_model_metrics(cdkformer, cdkformer_input)
    
    # 打印结果
    print("Model Comparison:")
    print("="*50)
    print("ResNetFD Metrics:")
    for k, v in resnet_fd_metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("\nResNet Metrics:")
    for k, v in resnet_metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    
    print("\nCDKFormer Metrics:")
    for k, v in cdkformer_metrics.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")

if __name__ == '__main__':
    compare_models()