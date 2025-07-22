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
from test_wipose import *
from getrideof import *
from models.wm import *
#from getrideof import *
#from PHASEupsamp import TerminalCSIInput2
model_path = 'weightsWM/'  # 指定保存模型的目录
model_name = 'wisppn-20251028.pkl'  # 指定模型的文件名
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True  # 启用cudnn.benchmark

from models.wisppn_resnet import *
batch_size = 36
num_epochs = 200

learning_rate = 0.001

def getMinibatch(file_names):
    file_num = len(file_names)
    csi_data = torch.zeros(file_num, 3, 30, 30)
    #csi_data_phase = torch.zeros(file_num, 3, 30, 30)

    #GT_paf_st1 = torch.zeros(file_num, 10, 36, 36)
    #GT_paf_st2 = torch.zeros(file_num, 12, 36, 36)
    #GT_paf_st3 = torch.zeros(file_num, 16, 36, 36)
    coords = torch.zeros(file_num, 1, 36)


    #GT_heatmap_st1 = torch.zeros(file_num, 6, 36, 36)
    #GT_heatmap_st2 = torch.zeros(file_num, 6, 36, 36)
    #GT_heatmap_st3 = torch.zeros(file_num, 7, 36, 36)
    

    for i in range(file_num):
        #print(file_names[i])
        #data = hdf5storage.loadmat(file_names[i], variable_names={'CSI', 'SkeletonPoints'})
        csi_data[i, :, :, :] = torch.from_numpy(PracticalCSIInput(file_names[i])).type(torch.FloatTensor)
        coords[i, :, :] = torch.from_numpy(readOppGTWM(file_names[i])).type(torch.FloatTensor)
        


    return csi_data, coords




# 初始化一个空列表来存储所有的.mat文件路径
mats = []
mats1 = []


#mats += glob.glob('/home/aistudio/data/data317616/group1_70_two_person_percent1/*.mat')
#mats += glob.glob('/home/aistudio/data/data317616/group1_70_two_person_percent2/*.mat')

mats += glob.glob('/home/qyy/notebooks/group1_70_percent_bend/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_crouch/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_lean/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_push/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_sit/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_stand/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_walk/*.mat')
mats += glob.glob('/home/qyy/notebooks/group1_70_percent_wave/*.mat')

#Selected_Mat_Files_bend.zip

#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_bend/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_crouch/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_lean/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_push/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_sit/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_stand/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_walk/*.mat')
#mats += glob.glob('/home/qyy/notebooks/Selected_Mat_Files_wave/*.mat')
shuffle(mats)
#print[mats]
# 现在mats包含了所有指定目录下的.mat文件路径
#mats=mats[0:1000]
mats_num = len(mats)
batch_num = int(np.floor(mats_num/batch_size))
# 加载预训练模型的权重
def load_partial_weights(model, model_path):
    """
    加载部分权重：只加载与模型结构匹配的权重
    """
    pretrained_dict = torch.load(model_path)
    model_dict = model.state_dict()

    # 过滤出匹配的权重
    matched_weights = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # 更新模型字典
    model_dict.update(matched_weights)
    model.load_state_dict(model_dict)

    print(f"Loaded {len(matched_weights)}/{len(pretrained_dict)} weights from {model_path}")
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

#wisppn = ResNet(ResidualBlock, [0, 0, 0 ,0])
# resnet = ResNet(ResidualBlock, [3, 4, 6, 3])
# resnet = ResNet(Bottleneck, [3, 4, 6, 3])
#wisppn = torch.load('weights2/65PCK5.pkl')
#wisppn = torch.load('weights2/wisppn-20251028-epoch55.pkl')
#torch.save(wisppn.state_dict(), "model.pt")
wisppn2 =WimoseNet()
#wisppn2.load_state_dict(torch.load("model.pt"))
#wisppn2 = load_partial_weights(wisppn, "model.pt")
#wisppn2 = torch.load('weights2/wisppn-20251028-epoch55.pkl')
# 加载预训练权重到新的网络结构中
#wisppn = freeze_weights(wisppn, 'models/half_pose.pth')
# 加载预训练模型
#wisppn = torch.load('weights2/65PCK5.pkl')
# 保存预训练模型的权重
#torch.save(wisppn.state_dict(), "model.pt")

# 初始化新模型
#wisppn2 = ResNetFourStage(ResidualBlock, [0, 0, 0, 0])

# 加载部分权重到新模型中
#wisppn2 = load_partial_weights(wisppn2, "model.pt")
wisppn = wisppn2.cuda()

criterion_L2 = nn.MSELoss().cuda()
optimizer = torch.optim.Adam(wisppn.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15, 30, 45, 60, 75,90,105,120,135,150,165,180], gamma=0.7)

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


    totalloss = 0
    totalloss1 = 0
    totalloss2 = 0
    totalloss3 = 0
    totalloss4 = 0
    #val_loss=eval_value
    #scheduler.step(val_loss)
    #scheduler.step()
    start = time.time()
    # shuffling dataset
    shuffle(mats)
    used =  mats
    loss_x = 0
    # in each minibatch
    print('进行到epoche',epoch_index,'/',num_epochs)
    # 创建一个内层进度条来显示每个 epoch 内的 batch 进度
    torch.save(wisppn, 'weights2/wisppn-20251028-epoch55.pkl')
    mats_num = len(used)
    batch_num = int(np.floor(mats_num/batch_size))

    for batch_index in range(batch_num):


        batch_start = time.time()

        if batch_index % 200== 0:
            print('进行到epoche',epoch_index,'/',num_epochs,'进行到batch', batch_index, '/', batch_num)
        if batch_index < batch_num:
            file_names = used[batch_index*batch_size:(batch_index+1)*batch_size]
        else:
            file_names = used[batch_num*batch_size:]

        csi_data, coords = getMinibatch(file_names)

        csi_data = Variable(csi_data.to(device, non_blocking=True))
        coords = Variable(coords.to(device, non_blocking=True))
        #nostage_GT_paf = Variable(nostage_GT_paf.to(device, non_blocking=True))
        #WGI = torch.cat([nostage_GT_heatmap, nostage_GT_paf], 1)
        #print(WGI.shape)




        Pred= wisppn(csi_data)
     
        
        coords = coords.squeeze(1)  # 移除中间的维度1

        # 计算MSE损失
        criterion = nn.MSELoss()
        loss = criterion(Pred, coords)

        #loss1 = criterion_L2(paf1, nostage_GT_paf)+criterion_L2(pcm1, nostage_GT_heatmap)
        
        #loss = loss1*4
        #loss = loss4*4
        #loss = criterion_L2(paf3, test_GT_paf_st3)+criterion_L2(paf2, test_GT_paf_st2)+criterion_L2(paf1, test_GT_paf_st1)+criterion_L2(pcm1, test_GT_heatmap_st1)+criterion_L2(pcm2, test_GT_heatmap_st2)+criterion_L2(pcm3, test_GT_heatmap_st3)
        #print(pcm2.shape)
        #print(nostage_GT_heatmap.shape)
        #loss = criterion_L2(paf3, nostage_GT_paf)+criterion_L2(pcm2, nostage_GT_heatmap)
        #loss = loss*4
        totalloss += loss
        totalloss1 += loss
        totalloss2 += loss
        totalloss3 += loss
        totalloss4 += loss

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
    avg_loss1 = totalloss1 / batch_num
    avg_loss2 = totalloss2 / batch_num
    avg_loss3 = totalloss3 / batch_num
    avg_loss4 = totalloss4 / batch_num


    endl = time.time()
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss: {avg_loss:.8f}')
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss1: {avg_loss1:.8f}')
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss2: {avg_loss2:.8f}')
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss3: {avg_loss3:.8f}')
    print(f'Epoch {epoch_index+1}/{num_epochs} completed. Average Loss4: {avg_loss4:.8f}')
    print(f'Costing time: {(endl-start)/60:.8f} minutes')

torch.save(wisppn, 'wisppn-20241027.pkl')