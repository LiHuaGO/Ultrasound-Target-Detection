import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionSubmodule(nn.Module):
    """
    通道注意力子模块 (CAS)
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttentionSubmodule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),  # 输入维度是 in_channels
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)   # 输出维度是 in_channels
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 全局平均池化和全局最大池化
        avg_out = self.avg_pool(x).view(x.size(0), -1)  # 形状: (batch_size, in_channels)
        max_out = self.max_pool(x).view(x.size(0), -1)  # 形状: (batch_size, in_channels)
        # 拼接并生成通道注意力权重
        channel_weights = self.sigmoid(self.mlp(avg_out) + self.mlp(max_out))
        channel_weights = channel_weights.unsqueeze(2).unsqueeze(3)  # 形状: (batch_size, in_channels, 1, 1)
        return channel_weights

class SpatialAttentionSubmodule(nn.Module):
    """
    空间注意力子模块 (SAS)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttentionSubmodule, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 在通道维度上进行平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # 拼接并生成空间注意力权重
        spatial_weights = self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return spatial_weights

class RMA(nn.Module):
    """
    残差从尺度注意力
    多膨胀率局部增强（创新点3）
    结合残差连接和多尺度感受野
    """

    def __init__(self, in_channels, dilations=[1, 3, 5]):
        super().__init__()
        self.branches = nn.ModuleList()
        for d in dilations:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3,
                              padding=d, dilation=d, groups=in_channels),
                    nn.BatchNorm2d(in_channels),
                    nn.GELU()
                )
            )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * len(dilations), in_channels, 1),
            nn.Dropout2d(0.1)
        )
        self.res_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        branch_outs = [branch(x) for branch in self.branches]
        concat_out = torch.cat(branch_outs, dim=1)
        fused = self.fusion(concat_out)
        return self.res_conv(identity + fused)

# 改进的超声特征增强注意力机制 (UFEA5浅层)
class UFEA(nn.Module):
    """
    创新点：
    1. 残差从尺度注意力 RMA 模块
    2. 交叉特征融合(CFF) - 在空间注意力前引入通道和空间特征的交叉融合
    """

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7, dilations=[1, 3, 5]):
        super(UFEA, self).__init__()
        self.cas = ChannelAttentionSubmodule(in_channels, reduction_ratio)

        self.rma = RMA(in_channels, dilations=dilations)  # 输入通道、膨胀率列表

        # 交叉特征融合
        self.cff_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True)
        )

        self.sas = SpatialAttentionSubmodule(kernel_size)

        # 初始化参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 通道注意力
        channel_weights = self.cas(x)
        x_channel = x * channel_weights

        # 交叉特征融合
        cff_input = torch.cat([x, x_channel], dim=1)
        x_cff = self.cff_conv(cff_input)

        x_rma = self.rma(x_cff)

        # 空间注意力
        spatial_weights = self.sas(x_rma)
        x_spatial = x_rma * spatial_weights

        # 最终残差连接（可选，根据需求调整）
        x_final = x + x_spatial  # 保持原始信息流
        return x_final


