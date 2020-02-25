# coding: utf-8

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim     
import sys
sys.path.append("..")           # 将当前目录的上级目录添加进文件搜索目录
from utils.utils import MyDataset, validate, show_confMat
from datetime import datetime

train_txt_path = os.path.join("..", "..", "Data", "train.txt")      # 训练集的地址
valid_txt_path = os.path.join("..", "..", "Data", "valid.txt")

classes_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

train_bs = 16
valid_bs = 16
lr_init = 0.001
max_epoch = 1

# log
result_dir = os.path.join("..", "..", "Result")

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')        #显示当前时间

log_dir = os.path.join(result_dir, time_str)    
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# -------------------------------------------- step 1/5 : 加载数据 -------------------------------------------

# 数据预处理设置
normMean = [0.4948052, 0.48568845, 0.44682974]      # 设置预处理时图像均值
normStd = [0.24580306, 0.24236229, 0.2603115]       # 设置方差 
normTransform = transforms.Normalize(normMean, normStd)     #  对数据按通道进行标准化，即先减均值，再除以标准差，注意是 chw
trainTransform = transforms.Compose([           # 由transform构成的列表.用于图像预处理时的一系列处理，这里依次是
    transforms.Resize(32),                      # 重置图像分辨率
    transforms.RandomCrop(32, padding=4),       # 依据给定的size随机裁剪
    transforms.ToTensor(),                      # 将PIL Image或者 ndarray 转换为tensor，并且归一化至[0-1]
    normTransform                           # 对数据按通道进行标准化
])

validTransform = transforms.Compose([
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)       
valid_data = MyDataset(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)    # shuffle = True打乱数据顺序
valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs)

# ------------------------------------ step 2/5 : 定义网络 ------------------------------------


class Net(nn.Module):         # 定义网络
    def __init__(self):             # 初始化函数
        super(Net, self).__init__()     # 初始化父类
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)       # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # 定义权值初始化
    def initialize_weights(self):       # 初始化数据
        for m in self.modules():        # 遍历每个模块分别初始化
            if isinstance(m, nn.Conv2d):       #   如果是nn.Conv2d模块 ，则执行下面的代码
                torch.nn.init.xavier_normal_(m.weight.data)     # 初始化，下划线表示在执行相应操作后，直接覆盖原来的数据
                if m.bias is not None:
                    m.bias.data.zero_()         # 初始化偏执为零
            elif isinstance(m, nn.BatchNorm2d): # 如果模块是nn.BatchNorm2d
                m.weight.data.fill_(1)          # 权重设置为1
                m.bias.data.zero_()             # 初始化偏执为零
            elif isinstance(m, nn.Linear):      # 如果是全连接层
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()


net = Net()     # 创建一个网络

# ================================ #
#        finetune 权值初始化
# ================================ #

# load params
pretrained_dict = torch.load('net_params.pkl')      # 加载预训练模型参数

# 获取当前网络的dict
net_state_dict = net.state_dict()

# 剔除不匹配的权值参数
pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}

# 更新新模型参数字典
net_state_dict.update(pretrained_dict_1)

# 将包含预训练模型参数的字典"放"到新模型中
net.load_state_dict(net_state_dict)

# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------
# ================================= #
#         按需设置学习率
# ================================= #

# 将fc3层的参数从原始网络参数中剔除
ignored_params = list(map(id, net.fc3.parameters()))
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

# 为fc3层设置需要的学习率
optimizer = optim.SGD([
    {'params': base_params},
    {'params': net.fc3.parameters(), 'lr': lr_init*10}],  lr_init, momentum=0.9, weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)     # 设置学习率下降策略

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------

for epoch in range(max_epoch):

    loss_sigma = 0.0    # 记录一个epoch的loss之和
    correct = 0.0
    total = 0.0
    scheduler.step()  # 更新学习率

    for i, data in enumerate(train_loader):
        # 获取图片和标签
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)

        # forward, backward, update weights
        optimizer.zero_grad()       # 梯度清空
        outputs = net(inputs)      # 前向传播
        loss = criterion(outputs, labels)   #计算误差
        loss.backward()             # 反向传播  
        optimizer.step()            # 更新参数

        # 统计预测信息
        _, predicted = torch.max(outputs.data, 1)   # 获取网络推测结构
        total += labels.size(0)                     # 统计总共参与推测的图片数量
        correct += (predicted == labels).squeeze().sum().numpy()    # 统计模型推测的正确的结果数量
        loss_sigma += loss.item()                   

        # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
        if i % 10 == 9:
            loss_avg = loss_sigma / 10
            loss_sigma = 0.0
            print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                epoch + 1, max_epoch, i + 1, len(train_loader), loss_avg, correct / total))
            print('参数组1的学习率:{}, 参数组2的学习率:{}'.format(scheduler.get_lr()[0], scheduler.get_lr()[1]))
    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    loss_sigma = 0.0
    cls_num = len(classes_name)
    conf_mat = np.zeros([cls_num, cls_num])  # 混淆矩阵
    net.eval()
    for i, data in enumerate(valid_loader):

        # 获取图片和标签
        images, labels = data
        images, labels = Variable(images), Variable(labels)

        # forward
        outputs = net(images)
        outputs.detach_() # 切断反向传播

        # 计算loss
        loss = criterion(outputs, labels)
        loss_sigma += loss.item()

        # 统计
        _, predicted = torch.max(outputs.data, 1)
        # labels = labels.data    # Variable --> tensor

        # 统计混淆矩阵
        for j in range(len(labels)):
            cate_i = labels[j].numpy()
            pre_i = predicted[j].numpy()
            conf_mat[cate_i, pre_i] += 1.0

    print('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))
print('Finished Training')

# ------------------------------------ step5: 绘制混淆矩阵图 ------------------------------------

conf_mat_train, train_acc = validate(net, train_loader, 'train', classes_name)
conf_mat_valid, valid_acc = validate(net, valid_loader, 'valid', classes_name)

show_confMat(conf_mat_train, classes_name, 'train', log_dir)
show_confMat(conf_mat_valid, classes_name, 'valid', log_dir)
