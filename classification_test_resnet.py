'''

对分类模型进行测试，同时比较和卷积核部分传输的性能区别

'''
import torch
import torchvision
from torchvision import models, transforms
import torch.nn as nn
import time
import os 

from classification_crop import CamExtractor_resnet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

start = time.time()
BATCH_SIZE = 64

test_data = torchvision.datasets.STL10('/home/lab239-5/users/liuchuanhong/CNN_classification/source/stl', 'test', 
        transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()]), download=True)
print(len(test_data))

test_loader = torch.utils.data.DataLoader(test_data, batch_size = BATCH_SIZE, shuffle=True)
print(len(test_loader))

criterion = nn.CrossEntropyLoss()   #交叉熵

# 导入模型
# chkpt_path = '/home/messor/users/liuchuanhong/liuchuanhong/CNN_classification/model/resnet_10_classes_channel_20dB.pth'   # resnet 
chkpt_path = '/home/lab239-5/users/liuchuanhong/CLUB-master/model/resnet18/resnet_model/resnet_10_classes_25db_mi_12.pth'
classes = 10

# 基于resnet18的分类
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, classes)


model.load_state_dict(torch.load(chkpt_path))
model.cuda()
model.eval()
# print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_name = 'layer4'  # resnet18

def find_importance(crop_ratio):
    f = open(r'/home/lab239-5/users/liuchuanhong/CNN_classification/source/Resnet18+MI/Resnet18_25dB.txt')
    content = f.read()#读进来的是和txt文档一样的
    # print(content)
    content_split = content.split()#按照空格把读取的txt文档做成list,list每一个值都是str
    content_split = list(map(int, content_split))#把列表的str转换成int
    # print(content_split)
    # print(len(content_split))
    f.close()
    feature_num = 512
    cls_num = 10 #一共有10类
    airplane = content_split[0:feature_num]
    bird = content_split[feature_num : feature_num*2]
    car = content_split[feature_num*2 : feature_num*3]
    cat = content_split[feature_num*3 : feature_num*4]
    deer = content_split[feature_num*4 : feature_num*5]
    dog = content_split[feature_num*5 : feature_num*6]
    horse = content_split[feature_num*6 : feature_num*7]
    monkey = content_split[feature_num*7 : feature_num*8]
    ship = content_split[feature_num*8 : feature_num*9]
    truck = content_split[feature_num*9 : feature_num*10]

    im_num = round(512*(1-crop_ratio)/10)
    # im_num = 36

    importance = airplane[0 : im_num] + bird[0 : im_num] + car[0 : im_num] + cat[0 : im_num]+ deer[0 : im_num
                ] + dog[0 : im_num] + horse[0 : im_num] + monkey[0 : im_num] + ship[0 : im_num] + truck[0 : im_num]

    print("initial length of importance: ", len(importance))
    importance = list(set(importance))

    print("actually length of importance: ", len(importance))
    return importance

crop_ratio = 0.8
index = find_importance(crop_ratio)

cam_extractor = CamExtractor_resnet(model, index, target_layer=layer_name)   # resnet

correct_test1 = 0.
total_test1 = 0.
loss_test1 = 0.
loss_test_img1 = list()
correct_test2 = 0.
total_test2 = 0.
loss_test2 = 0.
loss_test_img2 = list()
with torch.no_grad():
    for j, data in enumerate(test_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, crop_outputs = cam_extractor.forward_pass(inputs)
        loss1 = criterion(outputs, labels)
        loss2 = criterion(crop_outputs, labels)
        loss_test_img1.append(loss1.squeeze().cpu().numpy())
        loss_test_img2.append(loss2.squeeze().cpu().numpy())
        _, predicted1 = torch.max(outputs.data, 1)
        _, predicted2 = torch.max(crop_outputs.data, 1)
        total_test1 += labels.size(0)
        total_test2 += labels.size(0)
        correct_test1 += (predicted1 == labels).squeeze().cpu().sum().numpy()
        correct_test2 += (predicted2 == labels).squeeze().cpu().sum().numpy()
        loss_test1 += loss1.item()
        loss_test2 += loss2.item()

        if j%50 == 0:
            print(">>>>>>completed " + str(j) + "<<<<<<")
            print("Acc1:{:.2%}".format( correct_test1 / total_test1))
            print("Acc2:{:.2%}".format( correct_test2 / total_test2))

    loss_test_mean1 = loss_test1 / len(test_loader)
    loss_test_mean2 = loss_test2 / len(test_loader)

end = time.time()
print("Loss1: {:.4f} Acc1:{:.2%}".format( loss_test_mean1, correct_test1 / total_test1))
print("Loss2: {:.4f} Acc2:{:.2%}".format( loss_test_mean2, correct_test2 / total_test2))
print(end - start)
