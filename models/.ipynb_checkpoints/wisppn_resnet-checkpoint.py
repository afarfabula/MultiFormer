import scipy.io as sio
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
# from model import locNN
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from models.rtpose_vgg import get_model
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


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()

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
        BS = x.shape[0]
        #x = x.view(BS , 128 , 3 , 12, 36)
        x = x.contiguous().view(BS , 128 , 36, 36)

      
        
        xt = F.interpolate(x_original, scale_factor=(36/30,128/30), mode='bilinear', align_corners=False)
        xt = xt.permute(0, 3, 1, 2)
        xt = F.interpolate(xt, scale_factor=(12,1), mode='bilinear', align_corners=False)
        #xt = self.firstBN(x)
        
        #print('线性插值输出', x.shape)
        #x = x.permute(0, 3, 1, 2)
        #x = F.interpolate(x, scale_factor=(12,1), mode='bilinear', align_corners=False)
        x = self.firstBN(x)
        xt = self.firstBN2(xt)

        #print('线性插值输出', x.shape)
        #
        #x = self.layer1(x)
        #x = self.layer2(x)
        #x = self.layer3(x)
        #print('RESNET输出', x.shape)

       # x = self.layer4(x)
        x, _ = self.tf(x)
        xt,_ = self.tf2(xt)
        #print('注意力输出', x.shape)
        paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1 = self.opp(torch.max(x, xt),st1,st2)

        #
        #x = self.decode(x)
        return paf4,pcm4,paf3,pcm3,paf2,pcm2,paf1,pcm1