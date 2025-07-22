import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)
class ChannelMatch(nn.Module):
    def __init__(self, gate_channels, output_channels):
        super(ChannelMatch, self).__init__()
        # 定义中间维度的大小
        self.intermediate_channels = 256  # 这个值可以根据需要调整
        
        # 第一个线性层
        self.linear1 = nn.Linear(gate_channels, self.intermediate_channels)
        
        # 第二个线性层
        self.linear2 = nn.Linear(self.intermediate_channels, output_channels)

    def forward(self, x):
        # 应用第一个线性层和激活层
        x = F.relu(self.linear1(x))
        
        # 应用第二个线性层
        x = self.linear2(x)
        
        return x
class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
        self.ChanMactch =  ChannelMatch(gate_channels, 256)
    def forward(self, x):
        batchsize = x.shape[0] 
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw
        #print ('通道注意力权重',channel_att_sum.shape)
        #channel_att_sum = torch.rand(4,128)
        channel_att_sum = self.ChanMactch(channel_att_sum)
        #scale = torch.rand(batchsize,128,36,36)
        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand(batchsize,256,36,36)
        #print (scale.shape)
        return scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        #print('空间权重维度',scale.shape)
        return scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        #self.no_spatial=no_spatial
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        channel_weight = self.ChannelGate(x)
        
        spatial_weight = self.SpatialGate(x)
        return channel_weight*spatial_weight

if __name__ == "__main__":
    # 创建随机输入张量 (batch_size, channels, height, width)
    x = torch.rand(1, 17, 36, 36)
    
    # 测试CBAM模块
    cbam = CBAM(gate_channels=17)
    CSw = cbam(x)
    
    # 打印输入输出形状
    print(f"Input shape: {x.shape}")
    print(f"通道空间注意力 Output shape: {CSw.shape}")
    
    
