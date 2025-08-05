import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import math
import time
import sys
import glob
import hdf5storage
from random import shuffle
import time
import os
from test_wipose import evalue_all_batch
from getrideof import *
from getrideof import *
model_path = 'weights/'  # 指定保存模型的目录
model_name = 'wisppn-20241027.pkl'  # 指定模型的文件名



from models.wisppn_resnet import MultiFormer, ResidualBlock, Bottleneck
batch_size = 36
num_epochs = 100

learning_rate = 0.00023

def getMinibatch(file_names):
    file_num = len(file_names)
    csi_data = torch.zeros(file_num, 3, 30, 30)
    
    GT_paf_st1 = torch.zeros(file_num, 10, 36, 36)
    GT_paf_st2 = torch.zeros(file_num, 12, 36, 36)
    GT_paf_st3 = torch.zeros(file_num, 16, 36, 36)


    GT_heatmap_st1 = torch.zeros(file_num, 6, 36, 36)
    GT_heatmap_st2 = torch.zeros(file_num, 6, 36, 36)
    GT_heatmap_st3 = torch.zeros(file_num, 7, 36, 36)

    for i in range(file_num):
        #print(file_names[i])
        #data = hdf5storage.loadmat(file_names[i], variable_names={'CSI', 'SkeletonPoints'})
        csi_data[i, :, :, :] = torch.from_numpy(PracticalCSIInput(file_names[i])).type(torch.FloatTensor)
         # 假设 OppMatProcess 返回一个包含两个 NumPy 数组的元组 (PAF, heatmap)
        GT_paf_st1[i, :, :, :], GT_heatmap_st1[i, :, :, :] = OppMatProcess_ST1(file_names[i])
        GT_paf_st2[i, :, :, :], GT_heatmap_st2[i, :, :, :] = OppMatProcess_ST2(file_names[i])
        GT_paf_st3[i, :, :, :], GT_heatmap_st3[i, :, :, :] = OppMatProcess_ST3(file_names[i])
        
                
    return csi_data, GT_paf_st1, GT_heatmap_st1,GT_paf_st2, GT_heatmap_st2,GT_paf_st3, GT_heatmap_st3




# 初始化一个空列表来存储所有的.mat文件路径
mats = []

# 将每个目录下的.mat文件路径添加到mats列表中
mats += glob.glob('/mnt/workspace/group1_70_percent1/*.mat')
mats += glob.glob('/mnt/workspace/group1_70_percent2/*.mat')
mats += glob.glob('/mnt/workspace/group1_70_percent3/*.mat')
mats += glob.glob('/mnt/workspace/group1_70_percent4/*.mat')
mats += glob.glob('/mnt/workspace/group1_70_percent5/*.mat')

# 现在mats包含了所有指定目录下的.mat文件路径
#mats=mats[0:10]
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))
# 加载预训练模型的权重
def load_partial_weights(model, model_path):
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model
def freeze_weights(model, pretrained_path):
    # 加载预训练权重
    model = load_partial_weights(model, pretrained_path)
    
    # 冻结加载的权重
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any([pretrained_path in n for n in model.state_dict().keys()]):
                print(f"Freezing {name}")
                param.requires_grad = False
    return model

wisppn = MultiFormer(ResidualBlock, [0, 0, 0 ,0])
# resnet = ResNet(ResidualBlock, [3, 4, 6, 3])
# resnet = ResNet(Bottleneck, [3, 4, 6, 3])
wisppn = torch.load('weights/epochsave.pkl')
# 加载预训练权重到新的网络结构中
#wisppn = freeze_weights(wisppn, 'models/half_pose.pth')
wisppn = wisppn.cuda()

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75,90], gamma=0.7)

wisppn.train()
alltime_loss=[]


for epoch_index in range(num_epochs):
    #evalue_all_batch(test_mats, wisppn)
 #eval_value = evalue_all_batch(test_mats, wisppn)


    # 在每个epoch结束后保存模型
    # 每10个epoch结束后保存模型
    if epoch_index % 11 == 0:
        try:
            # 构建完整的文件路径
            epoch_model_path = os.path.join(model_path, f'{model_name[:-4]}-epoch{epoch_index}.pkl')
            torch.save(wisppn, epoch_model_path)
            print(f"Model saved successfully for epoch {epoch_index} at {epoch_model_path}")
        except Exception as e:
            print(f"Failed to save model for epoch {epoch_index}: {e}")


    totalloss=0
    #val_loss=eval_value
    #scheduler.step(val_loss)
    #scheduler.step()
    start = time.time()
    # shuffling dataset
    shuffle(mats)
    loss_x = 0
    # in each minibatch
    print('进行到epoche',epoch_index,'/',num_epochs)
    # 创建一个内层进度条来显示每个 epoch 内的 batch 进度
    torch.save(wisppn, 'weights/epochsave.pkl')
    
    for batch_index in range(batch_num):
        
        
        batch_start = time.time()
       
        if batch_index % 100 == 0:
            print('进行到epoche',epoch_index,'/',num_epochs,'进行到batch', batch_index, '/', batch_num)
        if batch_index < batch_num:
            file_names = mats[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            file_names = mats[batch_num*batch_size:]

        csi_data, GT_paf_st1, GT_heatmap_st1,GT_paf_st2, GT_heatmap_st2,GT_paf_st3, GT_heatmap_st3 = getMinibatch(file_names)

        csi_data = Variable(csi_data.cuda())
        #print('csi_data输入维度',csi_data.shape)
        

        #test_GT_paf,test_GT_heatmap = OppMatProcess( mat_file = 'newtrain/test1.mat')
        test_GT_paf_st1 = Variable(GT_paf_st1.cuda())
        test_GT_heatmap_st1 = Variable(GT_heatmap_st1.cuda())

        test_GT_paf_st2 = Variable(GT_paf_st2.cuda())
        test_GT_heatmap_st2 = Variable(GT_heatmap_st2.cuda())

        test_GT_paf_st3 = Variable(GT_paf_st3.cuda())
        test_GT_heatmap_st3 = Variable(GT_heatmap_st3.cuda())


        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1= wisppn(csi_data, torch.cat([test_GT_paf_st1, test_GT_heatmap_st1], 1),torch.cat([test_GT_paf_st2,test_GT_heatmap_st2,test_GT_paf_st1, test_GT_heatmap_st1], 1))
        # 将列表中的张量合并成一个单一的张量
 
        #print('pcm输出维度',pcm3.shape,pcm2.shape,pcm1.shape)
        #print('paf输出维度',paf3.shape,paf2.shape,paf1.shape)
        #loss = criterion_L2(paf4, test_GT_paf)+criterion_L2(paf3, test_GT_paf)+criterion_L2(paf2, test_GT_paf)+criterion_L2(paf1, test_GT_paf)+criterion_L2(pcm1, test_GT_heatmap)+criterion_L2(pcm2, test_GT_heatmap)+criterion_L2(pcm3, test_GT_heatmap)+criterion_L2(pcm4, test_GT_heatmap)
        loss = criterion_L2(paf3, test_GT_paf_st3)+criterion_L2(paf2, test_GT_paf_st2)+criterion_L2(paf1, test_GT_paf_st1)+criterion_L2(pcm1, test_GT_heatmap_st1)+criterion_L2(pcm2, test_GT_heatmap_st2)+criterion_L2(pcm3, test_GT_heatmap_st3)
       
        
        totalloss += loss

        #print(loss.item())
        optimizer.zero_grad()
        #optimizer.zero_grad()  # 清空梯度
        loss.backward()
        optimizer.step()  # 先执行优化器的 step 方法
       # scheduler.step()  # 然后执行学习率调度器的 step 方法
        #batch_endl = time.time()
        #print('Costing time:', (batch_endl -batch_start) / 60)
        # 更新内层进度条的描述，显示当前的损失
        # 更新进度条

    
    # 在每个 epoch 结束后，你可以在这里添加验证步骤或其他操作
       # Update the progress bar with the average loss for this epoch
    avg_loss = totalloss / batch_num


    endl = time.time()
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss: {avg_loss:.8f}')
    print(f'Costing time: {(endl-start)/60:.8f} minutes')

torch.save(wisppn, 'wisppn-20241027.pkl')
