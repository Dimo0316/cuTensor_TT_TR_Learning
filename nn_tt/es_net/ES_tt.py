import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import tensorly as tl
import tensorly
from itertools import chain
from tensorly import unfold
from tensorly.decomposition import *
from scipy.linalg import svd
from scipy.linalg import norm
import matplotlib.pyplot as plt
import os
import numpy as np
import time

from torch.autograd import Variable




print('==> Loading data..')

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
'''
def sig(x):
	print("activate!")
	return torch.sigmoid(x)
'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        #x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x)        
        return output

def tt(X,rank):

    U=[0 for x in range(0,2)]
    B=[0 for x in range(0,1)]
    x_mat = unfold(X,0)
    U_,_,_=svd(x_mat)
    U[0]=U_[:,:rank[0]]
        
    x_mat = unfold(X,1)
    U_,_,_=svd(x_mat)
    U[1]=U_[:,:rank[1]]
    U[0]=torch.(U[0])
    U[1]=torch.from_numpy(U[1])
    
    B[0] = tl.tenalg.multi_mode_dot(X,(U[0],U[1]),[0,1],transpose=True)

    return U[0],U[1],B[0]

def tt_decomposition_fc_layer(layer, rank):
    l,r,core = tt(layer.weight.data, rank=rank)
    print(core.shape,l.shape,r.shape)
            
    right_layer = torch.nn.Linear(r.shape[0], r.shape[1])
    core_layer = torch.nn.Linear(core.shape[1], core.shape[0])
    left_layer = torch.nn.Linear(l.shape[1], l.shape[0])
    
    left_layer.bias.data = layer.bias.data
    left_layer.weight.data = l
    right_layer.weight.data = r.T
    core_layer.weight.data = core.T  #这句如果不加几乎无影响(暂时)

    new_layers = [right_layer, core_layer, left_layer]
    return nn.Sequential(*new_layers)

def gen_noises(model,  layer_ids, std=1, co_matrices=None):
    noises = []
    for i, param in enumerate(model.parameters()):
        if i in layer_ids:
            if co_matrices == None:
                noises.append(torch.randn_like(param) * std) #生成与 param shape一样的随机tensor
            else:
                sz = co_matrices[i].shape[0]
                m = MultivariateNormal(torch.zeros(sz), co_matrices[i])
                noise = m.sample()
                noises.append(noise.reshape(param.shape))
        else:
            noises.append(torch.zeros_like(param))
        noises[-1] = noises[-1].to(device)
    return noises

def es_update(model, epsilons, ls, lr, layer_ids, mode=1, update=True):
    #         模型， 随机出来的tensor，根据随机tensor计算的 loss，学习率,层数【0~9】， mode=2 , updata = True
    device = epsilons[0][0].device
    num_directions = len(epsilons) #40
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)  #8

    ls = torch.tensor(ls).to(device)
    if mode == 1:
        weight = ls
    else:
        weight = 1 / (ls + 1e-8)
    indices = torch.argsort(weight)[-elite_num:]
    mask = torch.zeros_like(weight)
    mask[indices] = 1

    weight *= mask
    weight = weight / torch.sum(weight)

    grad = []
    for l in epsilons[0]:
        grad.append(torch.zeros_like(l))

    for idx in indices:
        for i, g in enumerate(epsilons[idx]):
            grad[i] += g * weight[idx]
    if update:
        if mode==1:
            i = 0
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    param.requires_grad = False
                    param -= lr * g
                    param.requires_grad = True
                i += 1
        else:
            i = 0
            # print(len(grad), layer_ids)
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    # print("update")
                    param.requires_grad = False
                    param += lr * g
                    param.requires_grad = True
                i += 1

    return grad



def add_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param += noise
            param.requires_grad = True
        i += 1



def remove_noises(model, noises, layer_ids):
    i = 0
    for param, noise in zip(model.parameters(), noises):
        if i in layer_ids:
            param.requires_grad = False
            param -= noise
            param.requires_grad = True
        i += 1

def explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, return_list, if_mirror):
    #               模型   输入[400,1,28,28] 标签：400 ，损失函数，[0~9]   None     用于写入文件  result  True
    ep_rt = []
    ls_rt = []

    epsilon = gen_noises(model, layer_ids, std=0.01, co_matrices=co_matrices) #epsilon len(layer_ids) 随机的tensor
    add_noises(model, epsilon, layer_ids) # 权重矩阵 = 权重矩阵 + 随机矩阵
    outputs = model(inputs) # 前向
    loss = criterion(outputs, targets).item()  #计算损失 （item-> 将tensor 转化为浮点数）
    remove_noises(model, epsilon, layer_ids)  # 权重矩阵 = 权重矩阵 - 随机矩阵

    ep_rt.append(epsilon.copy()) #随机矩阵 list
    ls_rt.append(loss)  # loss 的list

    if if_mirror:        
        for i in range(len(epsilon)): # 每个随机 tensor取数相反数
            epsilon[i] = -epsilon[i]
        add_noises(model, epsilon, layer_ids) #添加 随机数
        outputs = model(inputs)  #前向
        loss = criterion(outputs, targets).item() #计算loss
        remove_noises(model, epsilon, layer_ids) #移除随机数
        ep_rt.append(epsilon)
        ls_rt.append(loss)

    return ep_rt, ls_rt

def test(test_acc, best_acc, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        print('|', end='')
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            if batch_idx % 10 == 0:
                print('=', end='')
    acc = 100. * correct / total
    print('|', 'Accuracy:', acc, '% ', correct, '/', total)
    test_acc.append(correct / total)
    return max(acc, best_acc)



device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
cnttt = 0

tl.set_backend('pytorch')

tt_net = Net()
linear_layer=tt_net._modules['fc1']
#rank = min(linear_layer.weight.data.numpy().shape) //2
tt_net._modules['fc1']=tt_decomposition_fc_layer(linear_layer, [64,64])

print(tt_net)
print("tt_net have {} paramerters in total".format(sum(x.numel() for x in tt_net.parameters())))

#  对于tt_net 的训练
if torch.cuda.is_available():
    #tt_net.cuda()#将所有的模型参数移动到GPU上
    tt_net.cuda()

model = tt_net
num_epoch = 40
lr = 0.5
lr0 = lr
step_size = 3
gamma = 0.5
co_matrices = None
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
optimizer = optim.SGD(model.parameters(), lr=0.001) #优化函数
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
criterion = nn.CrossEntropyLoss() #损失函数
    # print(model.parameters())
num_layers = len(model.state_dict()) # 有 num_layers 层  10

train_acc = []
test_acc = []
best_acc = 0
global_mean = []
global_var = []

es_mode = 2
num_directions = 40
num_directions0 = num_directions

if_alternate = False
fall_cnt = 0
if_es = True
if_bp = False
if_mirror = True

layer_ids = list(range(num_layers))   # 0~9  一共是10层
num_directions = num_directions0   #40
if if_mirror:
    num_directions = num_directions // 2 #  变成  20
lr = lr0  #  0.5

for epoch in range(num_epoch):
    print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
    print("|", end="")
    train_loss = 0
    correct = 0
    total = 0            
    epoch_mean = []
    epoch_var = []

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if if_alternate:
            layer_ids = [layer_id]
            layer_id = (layer_id + 1) % num_layers
            print("if_alternate if perform!~~~")
        total += len(inputs)
        ls = []
        epsilons = []
        processes = []
        result = []

        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        inputs.requires_grad = True

        for _ in range(num_directions):
            epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, result, if_mirror)
            epsilons.extend(epsilon)  # 每次循环里面有两个 epsilon，一正，一个相反数
            ls.extend(loss)   # 每次也就有两个 loss
            for l in loss:
                train_loss += l

        es_grad = es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)

        outputs = model(inputs)
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()
        if batch_idx % 10 == 0:
            print('=', end='')
    print('|', 'Accuracy:', 100. * correct / total, '% ', correct, '/', total)
    best_acc = test(test_acc, best_acc, model)

    if epoch % step_size == 0 and epoch:
        lr *= gamma
        lr = max(lr, 0.0125)
        if epoch % (step_size * 2) == 0: 
            num_directions = max(int(num_directions/gamma), num_directions + 1)
        pass
    train_acc.append(correct / total)

    if epoch >= 2:
        if train_acc[-1] - train_acc[-2] < 0.01 and train_acc[-2] - train_acc[-3] < 0.01:
            fall_cnt += 1
        else:
            fall_cnt = 0

    print('Current learning rate: ', lr)
    print('Current num_directions: ', num_directions, "mirror" if if_mirror else "")

    print('Best training accuracy overall: ', best_acc)