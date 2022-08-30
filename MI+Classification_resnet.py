# MI + 基于Resnet的分类
import numpy as np
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torchvision
import time

from mi_estimators import CLUB

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

#set_seed(1)  # 设置随机种子
label_name = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}

# 参数设置
MAX_EPOCH = -1
BATCH_SIZE = 64
LR = 0.001
log_interval = 10
val_interval = 1
classes = 10
start_epoch = -1
lr_decay_step = 7
x_dim = 512
y_dim = 512
hidden_size = 15
loop_num = 10
training_steps = 30

Val_Accuracy = []
BEST_VAL_ACC = 0.
# ============================ step 1/5 数据 ============================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
#    transforms.Normalize(norm_mean, norm_std),
])

valid_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
#    transforms.Normalize(norm_mean, norm_std),
])

train_data = torchvision.datasets.STL10('/home/messor/users/liuchuanhong/liuchuanhong/CNN_classification/source/stl', 'train', transform=train_transform, download=True)
test_data = torchvision.datasets.STL10('/home/messor/users/liuchuanhong/liuchuanhong/CNN_classification/source/stl', 'test', transform=valid_transform, download=True)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
print("length of train_loader: ", len(train_loader))
print("length of valid_loader: ", len(valid_loader))


# ============================ step 2/5 模型 ============================
# 1/3 构建模型
resnet18_ft = models.resnet18(pretrained=True)

# 3/3 替换fc层
num_ftrs = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs, classes)
resnet18_ft.to(device)

resnet18_ft.to(device)
# chkpt_path = '/home/messor/users/liuchuanhong/liuchuanhong/CLUB-master/resnet_10_classes_15db_mi_10.pth'
# resnet18_ft.load_state_dict(torch.load(chkpt_path))
# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
# 法2 : conv 小学习率
optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略


# 构建MI互信息估计模型
mi_model = CLUB(x_dim, y_dim, hidden_size).cuda()
mi_optimizer = torch.optim.Adam(mi_model.parameters(), LR)
mi_est_values = []

# mi_chkpt_path = '/home/messor/users/liuchuanhong/liuchuanhong/CLUB-master/mi_model_resnet_15db.pth'
# mi_model.load_state_dict(torch.load(mi_chkpt_path))
# mi_model.eval()

for loop in range(loop_num):
    resnet18_ft.eval()
    for step in range(training_steps): 
        for i, data in enumerate(train_loader):
            mi_model.train()
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            x = inputs
            x = resnet18_ft.conv1(x)
            x = resnet18_ft.bn1(x)
            x = resnet18_ft.relu(x)
            x = resnet18_ft.maxpool(x)

            x = resnet18_ft.layer1(x)
            x = resnet18_ft.layer2(x)
            x = resnet18_ft.layer3(x)
            x = resnet18_ft.layer4(x)

            batch_x0 = resnet18_ft.avgpool(x)
            batch_x = batch_x0.view(batch_x0.size(0), -1)
            # channel
            snr = 25     #信噪比
            snr = 10**(snr/10.0)
            xpower = torch.sum(batch_x**2,1)/512.
            npower = xpower/snr
            noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
            noise = noise.normal_()*torch.sqrt(npower)
            noise = noise.transpose(1,0)
            batch_y = batch_x + noise
            
            model_loss = mi_model.learning_loss(batch_x, batch_y)
            # print("mi_model_loss: ", model_loss)
            
            mi_optimizer.zero_grad()
            model_loss.backward()
            mi_optimizer.step()
            mi_model.eval()
            mi_value = mi_model(batch_x, batch_y).item()
            print("mi_value: ", mi_value)
            # if i>75:
            #     mi_model.eval()
            #     # mi_est_values.append(mi_model(batch_x, batch_y).item())
            #     print(i,":   mutual information: ", mi_model(batch_x, batch_y).item(), end=" ")
            
            del batch_x, batch_y
            torch.cuda.empty_cache()
        
        if step%10 == 0:
            print("step: ", step)
    print('Find MI Model and Saving it...')
    # torch.save(mi_model.state_dict(),'mi_model_resnet_25db.pth')

    # ============================ step 5/5 训练 ============================

    mi_model.eval()
    print("start training...")
    for epoch in range(start_epoch + 1, MAX_EPOCH):

        loss_mean = 0.
        correct = 0.
        total = 0.

        resnet18_ft.train()
        for i, data in enumerate(train_loader):

            # forward
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # outputs = resnet18_ft(inputs)

            x = inputs
            x = resnet18_ft.conv1(x)
            x = resnet18_ft.bn1(x)
            x = resnet18_ft.relu(x)
            x = resnet18_ft.maxpool(x)

            x = resnet18_ft.layer1(x)
            x = resnet18_ft.layer2(x)
            x = resnet18_ft.layer3(x)
            x = resnet18_ft.layer4(x)

            batch_x0 = resnet18_ft.avgpool(x)
            batch_x = batch_x0.view(batch_x0.size(0), -1)
            
            # channel
            snr = 25    #信噪比
            snr = 10**(snr/10.0)  
            xpower = torch.sum(batch_x**2,1)/512.
            npower = xpower/snr
            noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
            noise = noise.normal_()*torch.sqrt(npower)
            # noise = noise.view(batch_x.size(0), -1)
            noise = noise.transpose(1,0)
            batch_y = batch_x + noise

            outputs = resnet18_ft.fc(batch_y)

            # backward
            optimizer.zero_grad()
            loss = criterion(outputs, labels) - 0.0001*mi_model(batch_x, batch_y)
            loss.backward()

            # update weights
            optimizer.step()

            # 统计分类情况
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).squeeze().cpu().sum().numpy()

            # 打印训练信息
            loss_mean += loss.item()
            if (i+1) % log_interval == 0:
                loss_mean = loss_mean / log_interval
                print("Training:Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, i+1, len(train_loader), loss_mean, correct / total))
                loss_mean = 0.


        scheduler.step()  # 更新学习率
        torch.cuda.empty_cache()

        # validate the model
        if (epoch+1) % val_interval == 0:

            correct_val = 0.
            total_val = 0.
            loss_val = 0.
            resnet18_ft.eval()
            with torch.no_grad():
                for j, data in enumerate(valid_loader):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    x = inputs
                    x = resnet18_ft.conv1(x)
                    x = resnet18_ft.bn1(x)
                    x = resnet18_ft.relu(x)
                    x = resnet18_ft.maxpool(x)

                    x = resnet18_ft.layer1(x)
                    x = resnet18_ft.layer2(x)
                    x = resnet18_ft.layer3(x)
                    x = resnet18_ft.layer4(x)

                    batch_x0 = resnet18_ft.avgpool(x)
                    batch_x = batch_x0.view(batch_x0.size(0), -1)
                    
                    # channel
                    snr = 25     #信噪比
                    snr = 10**(snr/10.0)  
                    xpower = torch.sum(batch_x**2,1)/512.
                    npower = xpower/snr
                    noise = torch.FloatTensor(512,batch_x.size(0)).to("cuda")
                    noise = noise.normal_()*torch.sqrt(npower)
                    # noise = noise.view(batch_x.size(0), -1)
                    noise = noise.transpose(1,0)
                    batch_y = batch_x + noise

                    outputs = resnet18_ft.fc(batch_y)

                    # outputs = resnet18_ft(inputs)
                    loss = criterion(outputs, labels) - 0.0001*mi_model(batch_x, batch_y)

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).squeeze().cpu().sum().numpy()

                    loss_val += loss.item()

                loss_val_mean = loss_val/len(valid_loader)
                print("Valid:\t Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(
                    epoch, MAX_EPOCH, j+1, len(valid_loader), loss_val_mean, correct_val / total_val))
            resnet18_ft.train()

            acc = correct_val / total_val
            Val_Accuracy.append(acc)
            # if acc > BEST_VAL_ACC:
            #     print('Find Better Model and Saving it...')
            #     torch.save(resnet18_ft.state_dict(),
            #             'resnet_10_classes_25db_mi_12.pth')
            #     BEST_VAL_ACC = acc
            #     print('Saved!')
    resnet18_ft.eval()
    if loop%5==0:
        print("loop: ", loop)


