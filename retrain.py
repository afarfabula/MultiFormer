import os
import glob
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from models.wisppn_resnet import ResNet, ResidualBlock
from test_wipose import evalue_all_batch  # 假设这些自定义函数已实现
from getrideof import *

# 配置参数
model_path = '0423weights/'
model_name = 'wisppn-20250424.pkl'
batch_size = 32
num_epochs = 200
learning_rate = 0.0001*0.5
num_workers = 8          # 根据 CPU 核心数调整
prefetch_factor = 2       # 预加载批次数量
lambda_jhm = 1.0         # 损失函数参数
beta_jhm = 1.0
lambda_paf = 0.3
beta_paf = 0.7

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # 启用cudnn.benchmark
def reorder_paf_channels(x, paf_indices):
    """
    参数:
        x: 输入张量 [batch, 38, H, W]
        paf_indices: 三重列表的通道顺序
    
    返回:
        重排后的张量 [batch, 38, H, W]
    """
    new_order = []
    for sublist in paf_indices:
        new_order.extend(sublist)
    
    if len(new_order) != 38 or sorted(new_order) != list(range(38)):
        raise ValueError("无效的paf_indices，必须包含0-37的所有唯一索引")
    
    return x.index_select(1, torch.tensor(new_order).to(x.device))

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
        nostage_GT_paf, nostage_GT_heatmap = OppMatProcess(file_path)
        #print(nostage_GT_paf.shape)
        
        #nostage_GT_heatmap = PcmReconstruct(pcm1, pcm2, pcm3)
        #nostage_GT_paf = torch.cat([GT_paf_st1, GT_paf_st2,GT_paf_st3], 1)
        nostage_GT_heatmap = nostage_GT_heatmap.squeeze(1)  # 移除第2维的1
        #print(nostage_GT_paf.shape)
        nostage_GT_paf = reorder_paf_channels(nostage_GT_paf.squeeze(1),paf_indices = [
    [0, 1, 6, 7, 12, 13, 20, 21, 28, 29],
    [2, 3, 8, 9, 14, 15, 22, 23, 30, 31, 32, 33],
    [4, 5, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 34, 35, 36, 37]
])
        #print(nostage_GT_paf.shape)
        return csi_data.float(), nostage_GT_heatmap.float(), nostage_GT_paf.float()

criterion_L2 = nn.MSELoss().cuda()

# 数据加载
def prepare_dataloaders():
    # 加载文件路径
    mats = []
    mats += glob.glob('/home/qyy/notebooks/output_label150_2024_11_30_*/group1_70_percent/*.mat')
    mats += glob.glob('/home/qyy/notebooks/0329dataset/output_label150_2025_03_29_*/*.mat')
    #mats += glob.glob('/home/qyy/notebooks/0329dataset/output_label150_2025_03_29_*_with_zeros/*.mat')
    
    # 分割数据集
    random.shuffle(mats)
    train_files = mats 
    
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
    #wisppn = ResNet(ResidualBlock, [1, 1, 1, 0]).to(device).float()
    #wisppn = torch.load('0423weights/wisppn-20250424.pkl')

    # Save the model as a .pth file
    #torch.save(wisppn, '0423weights/wisppn-20251028.pth')
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model
    wisppn = ResNet(ResidualBlock, [1, 1, 1, 0]).to(device).float()

    # Load the .pth file into the model
    wisppn.load_state_dict(torch.load('0423weights/wisppn-20250427-nt.pth'))
    
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
            heatmap = heatmap.to(device, non_blocking=True).squeeze(1)
            paf = paf.to(device, non_blocking=True).squeeze(1)
            
            # 前向传播
            with torch.cuda.amp.autocast(dtype=torch.float32): 
                paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = wisppn(csi,csi,csi)
            #print(paf4.shape)
            #print(paf.shape)
            
            
            # 计算损失
            loss1 = criterion_L2(paf1.squeeze(1), paf)+criterion_L2(pcm1.squeeze(1), heatmap)
            loss2 = criterion_L2(paf2.squeeze(1), paf)+criterion_L2(pcm2.squeeze(1), heatmap)
            loss3 = criterion_L2(paf3.squeeze(1), paf)+criterion_L2(pcm3.squeeze(1), heatmap)
            loss4 = criterion_L2(paf4.squeeze(1), paf)+criterion_L2(pcm4.squeeze(1), heatmap)
            loss = loss1+loss2+loss3+loss4
            
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
                      f'Loss: {loss.item():.8f} | Data Load Time: {data_loading_time:.2f}s')
                data_loading_time = 0.0
        
        # 学习率调整
        scheduler.step()
        torch.save(wisppn.state_dict(), os.path.join(model_path, f"{model_name[:-4]}.pkl"))
        # 保存模型
        if (epoch + 1) % 21 == 0:
            save_path = os.path.join(model_path, f"{model_name[:-4]}-epoch{epoch+1}.pkl")
            torch.save(wisppn.state_dict(), save_path)
            print(f"Model saved at {save_path}")
        
        # 打印epoch统计信息
        epoch_time = time.time() - start_epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1} Completed | Avg Loss: {avg_loss:.8f} | Time: {epoch_time:.2f}s')

if __name__ == "__main__":
    main()