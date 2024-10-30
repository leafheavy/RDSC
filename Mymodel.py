import os
import pandas as pd
import PIL.Image as Image
from tqdm import tqdm
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from RDSC import RDSC
    
# 将feat2根据feat1的形状进行变化
def feature_fusion(feat1, feat2, style='bilinear'):
    if(style == 'nearest'):
        feat2 = F.interpolate(feat2, size=feat1.size()[2:], mode='nearest') # 最邻近
    elif(style == 'bicubic'):
        feat2 = F.interpolate(feat2, size=feat1.size()[2:], mode='bicubic', align_corners=True) # 双三次
    else:
        feat2 = F.interpolate(feat2, size=feat1.size()[2:], mode='bilinear', align_corners=True) # 双线性
    fused_feat = torch.cat([feat1, feat2], dim=1)
    return fused_feat

def init_weight(m, style='kaiming'):
    if(style == 'Xavier'):
        print('Using Xavier init')
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)
    else:
        print('Using He init')
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias'):
                nn.init.constant_(m.bias, 0)

device = "cpu"
class MyModel(nn.Module):
    def __init__(self, num_class=9, num_diease=10, device='cpu'):
        super().__init__()
        self.device = device
        self.flc1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False, device=self.device)
        self.net1 = nn.Sequential(
            nn.BatchNorm2d(32),
            # SEBlock(channel=32, r=4, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # 32*112*112
            )
        self.flc2 = RDSC(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, device=self.device)
        self.net2 = nn.Sequential(
            nn.BatchNorm2d(64),
            # SEBlock(channel=64, r=3, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)) # 64*56*56
        
        # 学习“类别”
        self.fc_class = nn.Sequential(
            nn.Conv2d(96, 512, kernel_size=1, bias=False), # 512*14*14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)), # 512*1*1
            nn.Flatten(), # 将向量展平
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64,num_class))
        
        self.diease_block1_flc = RDSC(in_channels=96, out_channels=128, kernel_size=3, padding=1, stride=1, device=self.device)# RDSC(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1, device=self.device) # ,
        self.diease_block1 = nn.Sequential(
            nn.BatchNorm2d(128),
            # SEBlock(channel=128, r=4, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))# 128*28*28
        self.diease_block2_flc = RDSC(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, device=self.device)
        self.diease_block2 = nn.Sequential(
            nn.BatchNorm2d(256),
            # SEBlock(channel=256, r=4, device=self.device),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.diease_block3_flc = RDSC(in_channels=256, out_channels=512, kernel_size=1, device=self.device)
        self.diease_block3 = nn.Sequential(
            # SEBlock(channel=512, r=4, device=self.device),
            nn.BatchNorm2d(512),
            nn.ReLU()) # 512*7*7

        self.transform_conv = nn.Sequential(
            nn.Conv2d(992, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # 512*1*1
            nn.Flatten()# 将向量展平
            )

        self.is_diease = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Identity(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32,3)
            )
        
        self.fc_diease = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            # nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32,num_diease))
        
    def forward(self, x):
        x = x.to(torch.float32)
        map1 = self.net1(self.flc1(x)) # 32*112*112
        map2 = self.net2(self.flc2(map1)) # 64*56*56
        
        map_net = feature_fusion(map1, map2, style='bilinear') # 96*112*112
        
        class_head = self.fc_class(map_net)
        
        map3_dieaseblock1 = self.diease_block1(self.diease_block1_flc(map_net)) # 128*28*28
        map3_dieaseblock2 = self.diease_block2(self.diease_block2_flc(map3_dieaseblock1)) # 256*14*14
        map3_dieaseblock3 = self.diease_block3(self.diease_block3_flc(map3_dieaseblock2)) # 512*7*7

        map3 = feature_fusion(map3_dieaseblock1, map3_dieaseblock2, style='bilinear') # 384*28*28
        map3 = feature_fusion(map3, map3_dieaseblock3, style='bilinear') # 896*28*28

        map_diease = feature_fusion(map3, map_net, style='bilinear') # 992*28*28
        map_diease = self.transform_conv(map_diease)
    
        is_diease_head = self.is_diease(map_diease)
        diease_head = self.fc_diease(map_diease)

        return class_head, diease_head, is_diease_head
    
    # Re-parameterization
    def test(self, x):
        x = x.to(torch.float32)
        map1 = self.net1(self.flc1(x))
        # map1 = self.net1_1(self.flc1_1.test_dbb(x))
        map2 = self.net2(self.flc2.test_dbb(map1)) 
        map_net = feature_fusion(map1, map2, style='bilinear')

        class_head = self.fc_class(map_net) # Task1
        map3_dieaseblock1 = self.diease_block1(self.diease_block1_flc.test_dbb(map_net))
        map3_dieaseblock2 = self.diease_block2(self.diease_block2_flc.test_dbb(map3_dieaseblock1))
        map3_dieaseblock3 = self.diease_block3(self.diease_block3_flc.test_dbb(map3_dieaseblock2))
        
        map3 = feature_fusion(map3_dieaseblock1, map3_dieaseblock2, style='billinear')
        map3 = feature_fusion(map3, map3_dieaseblock3, style='billinear')
        
        map_diease = feature_fusion(map3, map_net, style='bilinear')
        map_diease = self.transform_conv(map_diease)

        is_diease_head = self.is_diease(map_diease)
        diease_head = self.fc_diease(map_diease)

        return class_head, diease_head, is_diease_head

diease_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
is_diease_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数
class_loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

mymodel = MyModel(device=device)
# mymodel.apply(init_weight)
mymodel = mymodel.to(device)

optimizer = optim.Adam(mymodel.parameters(), lr=0.001)  # 随机梯度下降优化器
epochs = 2
point_step = 2 # 模型预测的绘图的步长

# 读取包含图像名和标签的CSV文件
train_file = 'balanced_train.csv'  # 请将文件名替换为实际的CSV文件名
train_data = pd.read_csv(train_file)
train_data = train_data[:4]

class MyDataset(Dataset):
    def __init__(self, data, root_dir, transform=None):
        self.data = data
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name)
        labels = self.data.iloc[idx, 1:].values.astype(int)
        if self.transform:
            image = self.transform(image)
        return image, labels[0], labels[1], labels[2]

# 定义数据增强操作
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建Dataset实例
train_dataset = MyDataset(train_data, root_dir='pictures', transform=data_transforms)

# 创建DataLoader
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# attentionblock = TaskAttention([4, 9, 2]) # 由于是想获取三个任务之间的相对关系->不需要重复进行实例化->只需要参数不断更新即可
for epoch in tqdm(range(epochs)):
    mymodel.train()
    loss = 0

    for _, (inputs, diease_label, class_label, is_diease_label) in enumerate(train_loader):

        inputs = inputs.to(torch.float32)
        inputs = inputs.to(device)

        diease_label = diease_label.to(device)
        diease_label = diease_label.to(torch.long)

        class_label = class_label.to(device)
        class_label = class_label.to(torch.long)

        is_diease_label = is_diease_label.to(device)
        is_diease_label = is_diease_label.to(torch.long)

        class_outputs, diease_outputs, is_diease_outputs = mymodel(inputs)  # 将数据输入模型进行前向传播
        class_outputs = class_outputs.to(device)
        diease_outputs = diease_outputs.to(device)
        is_diease_outputs = is_diease_outputs.to(device)
        
        # attention_weights = (attentionblock(class_outputs, diease_outputs, is_diease_outputs)) * 3 # 引入任务注意力机制模块

        # 计算损失
        loss1 = diease_loss(diease_outputs, diease_label) 
        loss2 = class_loss(class_outputs, class_label)
        loss3 = is_diease_loss(is_diease_outputs, is_diease_label)

        # loss = attention_weights[0] * loss1 + attention_weights[1] * loss2 + attention_weights[2] * loss3
        loss = loss1 + loss2 + loss3
    
        optimizer.zero_grad()  # 梯度清零

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

filename = torch.load('parameter/RDSC+feature_fusion.pth', map_location=torch.device('cpu'))
mymodel.load_state_dict(filename)