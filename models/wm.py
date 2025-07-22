import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1,1)):
        super().__init__()
        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=(1,1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # 捷径路径（当通道数或尺寸变化时启用）
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if stride != (1,1) or in_channels != out_channels else None

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        return out

class WimoseNet(nn.Module):
    def __init__(self, input_shape=(3, 30, 30), output_dim=36):
        super().__init__()
        # 初始卷积适配 (3x30x30 -> 4x30x30)
        self.init_conv = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True)
        )
        
        # 残差块配置（根据表格参数严格对齐）
        self.res_blocks = nn.Sequential(
            # BLOCK1-2: 无下采样
            ResidualBlock(4, 4, stride=(1,1)),  # 4x30x30 -> 4x30x30
            ResidualBlock(4, 4, stride=(1,1)),
            
            # BLOCK3: 红色框内第一次下采样（表格中Output Size从30x20x4变为15x10x16）
            ResidualBlock(4, 16, stride=(2,2)),  # 4x30x30 ->16x15x15
            
            # BLOCK4-5 
            ResidualBlock(16, 16, stride=(1,1)), # 保持尺寸
            ResidualBlock(16, 32, stride=(2,2)), # 红色框内第二次下采样 ->32x7x7
            
            # BLOCK6-13: 无下采样
            *[ResidualBlock(32, 32) for _ in range(8)]  # 全部保持32x7x7
        )
        
        # 自适应池化解决最终特征图尺寸对齐
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 关键点回归网络（全连接层）
        self.regressor = nn.Sequential(
            nn.Linear(32, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 输出2*18=36
        )
        
    def forward(self, x):
        # 输入尺寸验证
        assert x.shape[1:] == (3, 30, 30), f"输入应为3x30x30, 实际得到{x.shape[1:]}"
        
        x = self.init_conv(x)        # [B,4,30,30]
        x = self.res_blocks(x)       # [B,32,7,7]
        x = self.adaptive_pool(x)    # [B,32,1,1]
        x = x.view(x.size(0), -1)    # [B,32]
        return self.regressor(x)     # [B,36]
if __name__ == "__main__":
    # 验证网络前向传播
    model = WimoseNet()
    dummy_input = torch.randn(2, 3, 30, 30)  # batch_size=2
    output = model(dummy_input)
    print(output.shape)  # torch.Size([2, 36])
