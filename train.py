import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from models.wisppn_resnet import ResNetFD, ResidualBlock
from test_wipose import evalue_all_batch  # 假设这些自定义函数已实现
from getrideof import PracticalCSIInput, OppMatProcess_ST1, OppMatProcess_ST2, OppMatProcess_ST3, PcmReconstruct

# 配置参数
model_path = 'weights3/'
model_name = 'wisppn-20251028.pkl'
batch_size = 70
num_epochs = 200
learning_rate = 0.00003
num_workers = 8          # 根据 CPU 核心数调整
prefetch_factor = 2       # 预加载批次数量
lambda_jhm = 1.0         # 损失函数参数
beta_jhm = 1.0
lambda_paf = 0.3
beta_paf = 0.7

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 启用cudnn.benchmark

# 自定义数据集
class CSIDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        
        # 加载单个样本
        csi_data = torch.from_numpy(PracticalCSIInput(file_path)).float()
        
        # 生成标签
        GT_paf_st1, pcm1 = OppMatProcess_ST1(file_path)
        GT_paf_st2, pcm2 = OppMatProcess_ST2(file_path)
        GT_paf_st3, pcm3 = OppMatProcess_ST3(file_path)
        nostage_GT_heatmap = PcmReconstruct(pcm1, pcm2, pcm3)
        nostage_GT_paf = torch.cat([GT_paf_st1, GT_paf_st2,GT_paf_st3], 1)
        
        return csi_data, nostage_GT_heatmap, nostage_GT_paf

# 加权MSE损失
def weighted_mse_loss(pred, gt, weights):
    return torch.mean(weights * (pred - gt).pow(2))

# 数据加载
def prepare_dataloaders():
    # 加载文件路径
    mats = []
    mats += glob.glob('/home/qyy/notebooks/group1_70_percent_*/*.mat')
    mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_*/*.mat')
    
    # 分割数据集
    random.shuffle(mats)
    train_files = mats + glob.glob('/home/qyy/notebooks/group1_70_percent_sit/*.mat')
    
    # 创建数据集
    train_dataset = CSIDataset(train_files)
    
    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor
    )
    
    return train_loader

# 主训练函数
def main():
    # 准备数据
    train_loader = prepare_dataloaders()
    
    # 初始化模型
    wisppn = ResNetFD(ResidualBlock, [1, 1, 1, 0]).to(device)
    
    # 定义优化器
    optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, 
        milestones=[15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180], 
        gamma=0.6
    )
    
    # 训练循环
    for epoch in range(num_epochs):
        wisppn.train()
        epoch_loss = 0.0
        
        # 数据加载计时
        data_loading_time = 0.0
        start_epoch = time.time()
        
        for batch_idx, (csi, heatmap, paf) in enumerate(train_loader):
            batch_start = time.time()
            
            # 异步数据传输
            csi = csi.to(device, non_blocking=True)
            heatmap = heatmap.to(device, non_blocking=True)
            paf = paf.to(device, non_blocking=True)
            
            # 前向传播
            paf_pred, heatmap_pred = wisppn(csi)
            
            # 计算权重
            weights_jhm = (lambda_jhm * heatmap.abs().pow(2) + beta_jhm).clamp(min=1e-6)
            weights_paf = (lambda_paf * paf.abs().pow(2) + beta_paf).clamp(min=1e-6)
            
            # 计算损失
            loss_paf = weighted_mse_loss(paf_pred, paf, weights_paf)
            loss_jhm = weighted_mse_loss(heatmap_pred, heatmap, weights_jhm)
            loss = (loss_paf + loss_jhm) * 4
            
            # 反向传播
            optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零
            loss.backward()
            optimizer.step()
            
            # 记录统计信息
            epoch_loss += loss.item()
            data_loading_time += time.time() - batch_start
            
            # 每100个batch打印一次
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Data Load Time: {data_loading_time:.2f}s')
                data_loading_time = 0.0
        
        # 学习率调整
        scheduler.step()
        
        # 保存模型
        if (epoch + 1) % 20 == 0:
            save_path = os.path.join(model_path, f"{model_name[:-4]}-epoch{epoch+1}.pkl")
            torch.save(wisppn.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        
        # 打印epoch统计信息
        epoch_time = time.time() - start_epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.4f} | Time: {epoch_time:.2f}s')

if __name__ == "__main__":
    main()