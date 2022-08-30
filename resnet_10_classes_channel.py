'''

基于resnet18的10分类模型  考虑信道部分

'''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("use device :{}".format(device))

#set_seed(1)  # 设置随机种子
label_name = {"airplane": 0, "bird": 1, "car": 2, "cat": 3, "deer": 4, "dog": 5, "horse": 6, "monkey": 7, "ship": 8, "truck": 9}

# 参数设置
MAX_EPOCH = 50
BATCH_SIZE = 32
LR = 0.001
log_interval = 10
val_interval = 1
classes = 10
start_epoch = -1
lr_decay_step = 7


# ============================ step 1/5 数据 ============================

#norm_mean = [0.4416781, 0.43651673, 0.40246016]
#norm_std = [0.25671667, 0.25384226, 0.2686348]

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

train_data = torchvision.datasets.STL10('stl', 'train', transform=train_transform, download=True)
test_data = torchvision.datasets.STL10('stl', 'test', transform=valid_transform, download=True)
# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)
print("length of train_loader: ", len(train_loader))
print("length of valid_loader: ", len(valid_loader))

# ============================ step 2/5 模型 ============================

# 1/3 构建模型
resnet18_ft = models.resnet18()

# 2/3 加载参数
#flag = 0
flag = 1
if flag:
    path_pretrained_model = "resnet18-5c106cde.pth"
    state_dict_load = torch.load(path_pretrained_model)
    resnet18_ft.load_state_dict(state_dict_load)

# 法1 : 冻结卷积层
flag_m1 = 0
# flag_m1 = 1
if flag_m1:
    for param in resnet18_ft.parameters():
        param.requires_grad = False
    print("conv1.weights[0, 0, ...]:\n {}".format(resnet18_ft.conv1.weight[0, 0, ...]))


# 3/3 替换fc层
num_ftrs = resnet18_ft.fc.in_features
resnet18_ft.fc = nn.Linear(num_ftrs, classes)
resnet18_ft.to(device)

# path_pretrained_model = "../model/resnet_10_classes_channel_0dB_2.pth"
# state_dict_load = torch.load(path_pretrained_model)
# resnet18_ft.load_state_dict(state_dict_load)


# ============================ step 3/5 损失函数 ============================
criterion = nn.CrossEntropyLoss()                                                   # 选择损失函数

# ============================ step 4/5 优化器 ============================
# 法2 : conv 小学习率
flag = 0
# flag = 1
if flag:
    fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是parameters的 内存地址
    base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
    optimizer = optim.SGD([
        {'params': base_params, 'lr': LR*0},   # 0
        {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)

else:
    optimizer = optim.SGD(resnet18_ft.parameters(), lr=LR, momentum=0.9)               # 选择优化器

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=0.1)     # 设置学习率下降策略


# ============================ step 5/5 训练 ============================
Val_Accuracy = []
BEST_VAL_ACC = 0.

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
        outputs = resnet18_ft(inputs)

        # backward
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
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

            if flag_m1:
              print("epoch:{} conv1.weights[0, 0, ...] :\n {}".format(epoch, resnet18_ft.conv1.weight[0, 0, ...]))

    scheduler.step()  # 更新学习率

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

                outputs = resnet18_ft(inputs)
                loss = criterion(outputs, labels)

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
        if acc > BEST_VAL_ACC:
            print('Find Better Model and Saving it...')
            torch.save(resnet18_ft.state_dict(),
                       '../model_correct_channel/resnet_10_classes_0db.pth')
            BEST_VAL_ACC = acc
            print('Saved!')
