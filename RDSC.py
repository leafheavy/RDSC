import torchvision.transforms as transforms
import torch
import torchvision.models as models
import torch.nn as nn

class GGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=False, device='cpu'):
        super(GGroup, self).__init__()
        self.device = device
        self.use_bias = use_bias
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=self.use_bias).to(self.device)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=self.use_bias).to(self.device)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1, bias=self.use_bias).to(self.device)
        self.weight = self.conv1.weight.to(self.device)
        if(self.use_bias):
            self.bias = self.conv1.bias.to(self.device)
        # 构造出ACNet结构
        # conv2_weight = self.conv2.weight.data
        self.conv2.weight.data[:, :, 0, :] = 0
        self.conv2.weight.data[:, :, 2, :] = 0
        # self.conv2.weight.data = conv2_weight
        self.conv2.weight[:, :, 0, :].grad = None
        self.conv2.weight[:, :, 2, :].grad = None

        # conv3_weight = self.conv3.weight.data
        self.conv3.weight.data[:, :, :, 0] = 0
        self.conv3.weight.data[:, :, :, 2] = 0
        # self.conv3.weight.data = conv3_weight
        self.conv3.weight[:, :, :, 0].grad = None
        self.conv3.weight[:, :, :, 2].grad = None

    # 推理的时候是一个分组卷积
    def transform_dbb(self):
        # 现在可以使用fused_conv来进行推理
        fused_weight = torch.zeros_like(self.conv1.weight).to(self.device)
        fused_weight = self.conv1.weight + self.conv2.weight + self.conv3.weight

        if(self.use_bias):
            fused_bias = torch.zeros_like(self.conv1.bias).to(self.device)
            fused_bias = self.conv1.bias + self.conv2.bias + self.conv3.bias

        # 创建一个新的Conv2d层，使用融合后的权重和偏置
        fused_conv = nn.Conv2d(self.conv1.in_channels, self.conv1.out_channels, kernel_size=3, stride=1, padding=1, bias=self.use_bias).to(self.device)
        fused_conv.weight = nn.Parameter(fused_weight).to(self.device)
        if(self.use_bias):
            fused_conv.bias = nn.Parameter(fused_bias).to(self.device)

        return fused_conv

    # 训练的时候是并行分组卷积
    def forward(self, x):

        branch1 = self.conv1(x)
        branch2 = self.conv2(x)
        branch3 = self.conv3(x)
        output = branch1 + branch2 + branch3
        return output

class Group2Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, groups, use_bias=False, device='cpu'):
        super(Group2Conv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.device = device
        self.use_bias = use_bias
        self.module_list = nn.ModuleList().to(self.device)
        
    # 推理的时候是等效成一个卷积
    def transform_dbb(self):
        # testmodel_list = nn.ModuleList()
        fused_conv = nn.Conv2d(self.in_channels, self.in_channels, self.kernel_size, 1, 1, bias=False, groups=self.groups).to(self.device)
        # 分组卷积的转换
        for g in range(self.groups):
            i = (self.in_channels // self.groups)*g

            # testmodel_list.append(self.module_list[g].transform_dbb())

            fused_conv.weight.data[i:i+(self.in_channels // self.groups)] = self.module_list[g].transform_dbb().weight.data
            if(self.use_bias):
                fused_conv.bias.data[i:i+(self.in_channels // self.groups)] = self.module_list[g].transform_dbb().bias.data

        return fused_conv #， testmodel_list
    
    # 训练的时候是分组卷积的并行结构
    def forward(self, x):
        x = x.to(self.device)
        out = torch.empty(x.shape[0], self.out_channels, x.shape[2], x.shape[3]).to(self.device)
        
        # 总共是self.groups个分组卷积
        for g in range(self.groups):
            expansion = self.in_channels // self.groups
            start = expansion*g
            end = (g + 1) * (self.out_channels) // self.groups
            input = x[:, start:end, :, :] # 这个才是对应的输入，input_channel=1
            
            # 多个分组卷积并行化
            if(len(self.module_list) <= self.groups): # 第一次进行并行化->加入list->list有self.groups个分组卷积
                conv = GGroup(expansion,  self.out_channels // self.groups, self.kernel_size, device=self.device, stride=1, padding=1) # 一个分组卷积的并行化
                self.module_list.append(conv)
                
            else: # 之后每个新的batch来，都沿用之前那些并行化卷积而不重新获取并行化卷积
                conv = self.module_list[g]
        
            branch = conv(input) # 计算每个分支
            out[:, start:end, :, :] = branch # 在通道维度上concat起来

        return out, self.module_list
    
# 自定义深度卷积
class RDSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, device='cpu', use_bias=False):
        super(RDSC, self).__init__()
        self.device = device
        # 逐深度卷积
        # self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=use_bias)
        # 对逐深度卷积进行DBB重构
        self.depthwise_conv = Group2Conv(in_channels, in_channels, kernel_size=3, groups=in_channels, device=self.device)
        # 逐点卷积
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=use_bias).to(self.device)

    def test_dbb(self, x):
        # 串联两个卷积
        # x = self.depthwise_conv(x)
        x = x.to(self.device)
        depth_dbb = self.depthwise_conv.transform_dbb() # 改进版本的深度可分离卷积
        x = depth_dbb(x)
        
        x = self.pointwise_conv(x)
        return x
    
    def forward(self, x):
        # 串联两个卷积
        # x = self.depthwise_conv(x)
        x, _ = self.depthwise_conv(x) # 改进版本的深度可分离卷积
        
        x = self.pointwise_conv(x)
        return x
    
