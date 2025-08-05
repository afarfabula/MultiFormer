import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.rtpose_vgg import*
import math
from ChannelTrans import ChannelTransformer

# 3x3 Convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1,  bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
#Fogery Detection论文复现
class ResNetFD(nn.Module):
    def __init__(self, block, layers):
        super(ResNetFD, self).__init__()

        self.in_channels = 3
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self.make_layer(block, self.in_channels, layers[0])
        self.layer2 = self.make_layer(block, 150, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 300, layers[3], 2)
        self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.opp = get_model_stage1()

        

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels*block.expansion):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels*block.expansion, stride=stride),
                nn.BatchNorm2d(out_channels*block.expansion))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels*block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        #print('初始', x.shape)
        x = F.interpolate(x, scale_factor=4.8, mode='bilinear', align_corners=False)
        #print('线性插值输出', x.shape)
        #
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print('RESNET输出', x.shape)

        #x = self.layer4(x)
        #x, _ = self.tf(x)
        #print('注意力输出', x.shape)
        paf1,pcm1 = self.opp(x,x,x)

        #
        #x = self.decode(x)
        return paf1,pcm1

# ResNet Module
class MultiFormer(nn.Module):
    def __init__(self, block, layers):
        super(MultiFormer, self).__init__()

        self.firstBN = nn.BatchNorm2d(128)
        self.firstBN2 = nn.BatchNorm2d(128)
        # 添加额外的标准化层
        self.finalBN = nn.BatchNorm2d(128)  # 用于标准化 x
        self.finalBN2 = nn.BatchNorm2d(128)  # 用于标准化 xt
        self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.tf2 = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.opp = get_model()





    def forward(self, x,st1,st2):

        x_original = x
        x = F.interpolate(x,  size=(128, 432), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3)
        BS = x.shape[0]
        x = x.contiguous().view(BS , 128 , 36, 36)



        xt = F.interpolate(x_original, size=(432, 128), mode='bilinear', align_corners=False)
        xt = xt.permute(0, 3, 1, 2)
        #xt = F.interpolate(xt, scale_factor=(12,1), mode='bilinear', align_corners=False)
        #xt = self.firstBN(x)
        xt = xt.contiguous().view(BS, 128, 36, 36)

        x = self.firstBN(x)
        xt = self.firstBN2(xt)

        x, _ = self.tf(x)
        xt,_ = self.tf2(xt)

        # 在 self.opp 调用之前再次标准化 x 和 xt
        x = self.finalBN(x)
        xt = self.finalBN2(xt)
        mean_x = torch.mean(x)
        mean_xt = torch.mean(xt)



        # 打印均值
        #print("均值 of x:", mean_x.item())
        #print("均值 of xt:", mean_xt.item())
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp( torch.cat([x, xt], 1),st1,st2)

        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1
class ResNetFourStage(nn.Module):
    def __init__(self, block, layers):
        super(ResNetFourStage, self).__init__()

        self.firstBN = nn.BatchNorm2d(128)
        self.firstBN2 = nn.BatchNorm2d(128)
        # 添加额外的标准化层
        self.finalBN = nn.BatchNorm2d(128)  # 用于标准化 x
        self.finalBN2 = nn.BatchNorm2d(128)  # 用于标准化 xt
        self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.tf2 = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.opp = get_modelST4()





    def forward(self, x,st1,st2):

        x_original = x
        x = F.interpolate(x,  size=(128, 432), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3)
        BS = x.shape[0]
        x = x.contiguous().view(BS , 128 , 36, 36)



        xt = F.interpolate(x_original, size=(432, 128), mode='bilinear', align_corners=False)
        xt = xt.permute(0, 3, 1, 2)
        #xt = F.interpolate(xt, scale_factor=(12,1), mode='bilinear', align_corners=False)
        #xt = self.firstBN(x)
        xt = xt.contiguous().view(BS, 128, 36, 36)

        x = self.firstBN(x)
        xt = self.firstBN2(xt)

        x, _ = self.tf(x)
        xt,_ = self.tf2(xt)

        # 在 self.opp 调用之前再次标准化 x 和 xt
        x = self.finalBN(x)
        xt = self.finalBN2(xt)
        mean_x = torch.mean(x)
        mean_xt = torch.mean(xt)



        # 打印均值
        #print("均值 of x:", mean_x.item())
        #print("均值 of xt:", mean_xt.item())
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp( torch.cat([x, xt], 1),st1,st2)

        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1
class MLP(nn.Module):
    def __init__(self, channelnum):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(128*3, 128)  # 第一个全连接层
        self.fc2 = nn.Linear(128, 128)    # 第二个全连接层
        self.fc3 = nn.Linear(128, 36*36)  # 第三个全连接层，输出大小与输入相同
        self.norm = nn.BatchNorm2d(channelnum)  # 添加 BatchNorm2d 归一化层

    def forward(self, x):
        # 将输入张量的后两个维度合并，使其变为 (batchsize, channelnum, 128*3)
        #x = x.view(x.size(0), x.size(1), -1)
        #print(x.shape)
        # 通过全连接层
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # 恢复为 (batchsize, channelnum, 36, 36)
        x = x.view(x.size(0), -1, 36, 36)
        
        # 添加归一化层
        x = self.norm(x)
        
        return x

class ResNet_PHASE(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_PHASE, self).__init__()

        self.firstBN = nn.BatchNorm2d(128)
        self.firstBN2 = nn.BatchNorm2d(128)
        self.firstBN3 = nn.BatchNorm2d(128)
        self.firstBN4 = nn.BatchNorm2d(128)
        self.MLP1 = MLP(12)
        self.MLP2 = MLP(12)
        
        self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.tf2 = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.opp = get_model()

        

    

    def forward(self, x,st1,st2,csi_data_phase):
        #print('初始', x.shape
        xf = csi_data_phase
        x_original = x
        x = F.interpolate(x, scale_factor=(128/30,216/30), mode='bilinear', align_corners=False)
        xf = F.interpolate(csi_data_phase, scale_factor=(128/30,216/30), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3)
        x = self.firstBN(x)
        #print(x.shape)
        xf  = xf.permute(0, 2, 1, 3)
        xf = self.firstBN2(x)
        x =torch.cat((x, xf), dim=-1) 
        
        
        BS = x.shape[0]
        #x = x.view(BS , 128 , 3 , 12, 36)
        xa = x.contiguous().view(BS , 128 , 36, 36)

      
        xtp = F.interpolate(csi_data_phase, scale_factor=(36/30,128/30), mode='bilinear', align_corners=False)
        xt = F.interpolate(x_original, scale_factor=(36/30,128/30), mode='bilinear', align_corners=False)
        xt = xt.permute(0, 3, 1, 2)
        xt = self.firstBN3(xt)
        xtp = xtp.permute(0, 3, 1, 2)
        xtp = self.firstBN4(xtp)
        #print(xtp.shape)
        #print(xtp.shape)
        
        xt =torch.cat((xt, xtp), dim=-1)
        #print(xt.shape)
        xta = F.interpolate(xt, scale_factor=(6,1), mode='bilinear', align_corners=False)
        xta = xta.contiguous().view(BS , 128 , 36, 36)
        #print(xta.shape)
        #xt = self.firstBN(x)
        ###
        
        
        
        
        
        ###
        #print('线性插值输出', x.shape)
        #x = x.permute(0, 3, 1, 2)
        #x = F.interpolate(x, scale_factor=(12,1), mode='bilinear', align_corners=False)
        #xa = self.firstBN(xa)
        #xta= self.firstBN2(xta)

        #print('线性插值输出', x.shape)
        #
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #print('RESNET输出', x.shape)

       # x = self.layer4(x)
        xa = self.MLP1(xa)
        xta = self.MLP1(xta)
        x, _ = self.tf(xa)
        xt,_ = self.tf2(xta)
        #print('注意力输出', x.shape)
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp( torch.cat([x, xt], 1),st1,st2)

        #
        #x = self.decode(x)
        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1


    
class ResNet_IR(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_IR, self).__init__()

        self.firstBN = nn.BatchNorm1d(128)
        self.firstBN2 = nn.BatchNorm1d(128)
        embed_dim = 128*3    # 嵌入维度 (N)
        num_heads = 3     # 注意力头数
        hidden_dim = 128  # 前馈网络隐藏层维度
        num_layers = 4   # 8 层注意力
        #self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        #self.tf2 = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.tf = Transformer(embed_dim, num_heads, hidden_dim, num_layers)
        self.tf2 = Transformer(embed_dim, num_heads, hidden_dim, num_layers)
        self.opp = get_model()

        

    

    def forward(self, x,st1,st2):
        #print('初始', x.shape)
        x_original = x
        #x = F.interpolate(x, scale_factor=(128/128,432/128), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3)
        #print('初始', x.shape)
        BS = x.shape[0]
        #x = x.view(BS , 128 , 3 , 12, 36)
        x = x.contiguous().view(BS, 128, 3*128)
      
        
        xt = x_original
        xt = xt.permute(0, 3, 1, 2)
        xt = xt.contiguous().view(BS , 128 , 3*128)

        x = self.firstBN(x)
        xt = self.firstBN2(xt)

        #print('注意力之前', x.shape)
        x= self.tf(x)
        #print('注意力输出', x.shape)
        xt = self.tf2(xt)
        
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp( torch.cat([x, xt], 1),st1,st2)

        #
        #x = self.decode(x)
        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1

class ResNet_IRNK(nn.Module):
    def __init__(self, block, layers):
        super(ResNet_IRNK, self).__init__()

        self.firstBN = nn.BatchNorm2d(128)
        self.firstBN2 = nn.BatchNorm2d(128)
        
        self.tf = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        self.tf2 = ChannelTransformer(vis=False, img_size=[36, 36], channel_num=128, num_layers=1, num_heads=3)
        
        self.opp = get_model()

        

    

    def forward(self, x,st1,st2):
        #print('初始', x.shape)
        x_original = x
        x = F.interpolate(x, scale_factor=(128/30,432/30), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 1, 3)
        #print('初始', x.shape)
        BS = x.shape[0]
        #x = x.view(BS , 128 , 3 , 12, 36)
        x = x.contiguous().view(BS, 128, 36,36)
      
        
        xt = x_original
        xt = F.interpolate(xt, scale_factor=(432/30,128/30), mode='bilinear', align_corners=False)
        xt = xt.permute(0, 3, 1, 2)
        xt = xt.contiguous().view(BS , 128 , 36,36)
        #print('初始', x.shape)
        
    
        x = self.firstBN(x)
        #print('初始', x.shape)
        xt = self.firstBN2(xt)

        #print('注意力之前', x.shape)
        x,_= self.tf(x)
        #print('注意力输出', x.shape)
        xt,_ = self.tf2(xt)
        
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp( torch.cat([x, xt], 1),st1,st2)

        #
        #x = self.decode(x)
        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1