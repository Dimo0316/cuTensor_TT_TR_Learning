import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR

import torchvision
import torchvision.transforms as transforms
from torchvision import models

import  matplotlib.pyplot as plt
import tensorly as tl
import tensorly
from itertools import chain
from tensorly.decomposition import parafac, partial_tucker, tucker

import os
import matplotlib.pyplot as plt
import numpy as np
import time
from torch.autograd import Variable
from torch.distributions.multivariate_normal import MultivariateNormal

import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Manager


################################## 1. load data #########################################################
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=400, shuffle=True, num_workers=0)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
#########################################################################################################

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

cnttt = 0

class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(
                        in_channels=1,
                        out_channels=16,
                        kernel_size=5,
                        stride=1,
                        padding=2)
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.out = nn.Linear(32 * 7 * 7, 10)
        # self.conv1.parameters.

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        for l in self.layers:
            if isinstance(l, nn.Linear):
                nn.init.xavier_normal_(l.weight.data)

    def forward(self, x):
        y = x.reshape(-1, 784)
        output = self.layers(y)
        return output


class Optimizer():
    def __init__(self, lr, momentum, step_size, gamma) -> None:
        self.lr = lr
        self.momentum = momentum
        self.step_size = step_size
        self.gamma = gamma
        self.step = 0
    
    def step(self, gamma):
        self.step += 1


def tucker_decomposition_conv_layer(layer, ranks):
    core, [last, first]  = partial_tucker(layer.weight.data, modes=[0, 1], rank=ranks, init='svd')
    # core, factors = partial_tucker(layer.weight.data, modes=[0, 1, 2, 3], rank=ranks, init='svd')
    #print(core.shape, last.shape, first.shape)

    # A pointwise convolution that reduces the channels from S to R3
    first_layer = torch.nn.Conv2d(in_channels=first.shape[0],
                out_channels=first.shape[1], kernel_size=1,
                stride=1, padding=0, dilation=layer.dilation, bias=False)

    # A regular 2D convolution layer with R3 input channels
    # and R3 output channels
    core_layer = torch.nn.Conv2d(in_channels=core.shape[1],
                out_channels=core.shape[0], kernel_size=layer.kernel_size,
                stride=layer.stride, padding=layer.padding, dilation=layer.dilation, bias=False)

    # A pointwise convolution that increases the channels from R4 to T
    last_layer = torch.nn.Conv2d(in_channels=last.shape[1], \
                out_channels=last.shape[0], kernel_size=1, stride=1,
                padding=0, dilation=layer.dilation, bias=True)

    last_layer.bias.data = layer.bias.data

    first_layer.weight.data = torch.transpose(first, 1, 0).unsqueeze(-1).unsqueeze(-1)
    last_layer.weight.data = last.unsqueeze(-1).unsqueeze(-1)
    core_layer.weight.data = core

    new_layers = [first_layer, core_layer, last_layer]
    # for i, l in enumerate(new_layers):
    #    print(i, l.weight.data.shape)
    return nn.Sequential(*new_layers)


# build model
def build(decomp=True):
    print('==> Building model..')
    tl.set_backend('pytorch')
    full_net = CNNNet()
    full_net = full_net.to(device)
    torch.save(full_net, 'model')
    if decomp:
        decompose()
    net = torch.load("model").to(device)
    print('==> Done')
    return net


# testing
def test(test_acc, best_acc, model):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    with True:
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


# decomposition
def decompose():
    model = torch.load("model").to(device)
    model.eval()
    model.cpu()
    layers = model._modules
    for i, key in enumerate(layers.keys()):
        if i >= len(layers.keys()):
            break
        if isinstance(layers[key], torch.nn.modules.conv.Conv2d):
            conv_layer = layers[key]
            rank = max(conv_layer.weight.data.numpy().shape) // 10
            ranks = [max(int(np.ceil(conv_layer.weight.data.numpy().shape[0] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[1] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[2] / 3)), 1),
                     max(int(np.ceil(conv_layer.weight.data.numpy().shape[3] / 3)), 1)]
            layers[key] = tucker_decomposition_conv_layer(conv_layer, ranks)
        torch.save(model, 'model')
    return model


# testing
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


def run_train(num_epoch, model):
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_acc = []
    test_acc = []
    best_acc = 0

    for epoch in range(num_epoch):
        # model.train()
        print('\nEpoch: ', epoch)
        print("|", end="")
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('=', end='')

        print('|', 'Accuracy:', 100. * correct / total,'% ', correct, '/', total)
        best_acc = test(test_acc, best_acc, model)

        train_acc.append(correct / total)
        scheduler.step()
        print('Current learning rate: ', scheduler.get_last_lr()[0])

    print('Best training accuracy overall: ', best_acc)
    return train_acc, test_acc, best_acc


def run_test(model):
    test_acc = []
    best_acc = 0
    best_acc = test(test_acc, best_acc, model)
    print('Testing accuracy: ', best_acc)
    return test_acc, best_acc


############################ functions for ES ######################################
def gen_noises(model,  layer_ids, std=1, co_matrices=None):
    noises = []
    for i, param in enumerate(model.parameters()):
        if i in layer_ids:
            if co_matrices == None:
                noises.append(torch.randn_like(param) * std)
            else:
                sz = co_matrices[i].shape[0]
                m = MultivariateNormal(torch.zeros(sz), co_matrices[i])
                noise = m.sample()
                noises.append(noise.reshape(param.shape))
        else:
            noises.append(torch.zeros_like(param))
        noises[-1] = noises[-1].to(device)
    return noises


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


def clone_params(model):
    params = []
    for param in model.parameters():
        params.append(torch.clone(param))
    return params


def es_update(model, epsilons, ls, lr, layer_ids, mode=1, update=True):
    device = epsilons[0][0].device
    num_directions = len(epsilons)
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)

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


def cma_es_update(model, epsilons, ls, layer_ids, params, mode=1, update=True):
    alpha_miu, alpha_sigma, alpha_cp, alpha_c1, alpha_clambda = params[:5]
    damping_factor = params[5]
    ps_sigma = params[6]
    co_matrices = params[7]
    sigmas = params[8]
    ps_c = sigmas[9]

    device = epsilons[0][0].device
    num_directions = len(epsilons)
    elite_rate = 0.2
    elite_num = max(int(elite_rate * num_directions), 1)

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
    delta = []
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
                    param -= alpha_miu * g
                    delta.append(-alpha_miu * g)
                    param.requires_grad = True
                else:
                    delta.append(g)
                i += 1
        else:
            i = 0
            # print(len(grad), layer_ids)
            for g, param in zip(grad, model.parameters()):
                if i in layer_ids:
                    # print("update")
                    param.requires_grad = False
                    param += alpha_miu * g
                    delta.append(alpha_miu * g)
                    param.requires_grad = True
                else:
                    delta.append(g)
                i += 1

    for i, g in enumerate(grad):
        if i in layer_ids:
            ps_sigma[i] = (1 - alpha_sigma) * ps_sigma[i] + torch.sqrt(alpha_sigma * (2 - alpha_sigma) * elite_num) * (1 / torch.sqrt(co_matrices[i])) * (delta[i] / sigmas[i])
            sigmas[i] = sigmas[i] * torch.exp(alpha_sigma / damping_factor * (torch.sqrt(torch.square(ps_sigma[i]).sum()) / np.sqrt(co_matrices[i].shape[0]) - 1))
            ps_c[i] = (1 - alpha_cp) * ps_c[i] + torch.sqrt(alpha_cp * (2 - alpha_cp) * elite_num) * (delta / sigmas[i])
            temp = 0
            for idx in indices:
                temp += torch.matmul(epsilons[idx][i], epsilons[idx][i].T)
            co_matrices[i] = (1 - alpha_clambda - alpha_c1) * co_matrices[i] + alpha_c1 * torch.matmul(ps_c[i], ps_c[i].T) + (1 / elite_num) * alpha_clambda * temp
    return grad, (ps_sigma, sigmas, ps_c, co_matrices)


def explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, cwd, return_list, if_mirror):
    ep_rt = []
    ls_rt = []

    epsilon = gen_noises(model, layer_ids, std=0.01, co_matrices=co_matrices)
    add_noises(model, epsilon, layer_ids)
    outputs = model(inputs)
    loss = criterion(outputs, targets).item()
    remove_noises(model, epsilon, layer_ids)

    ep_rt.append(epsilon.copy())
    ls_rt.append(loss)

    if if_mirror:        
        for i in range(len(epsilon)):
            epsilon[i] = -epsilon[i]
        add_noises(model, epsilon, layer_ids)
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()
        remove_noises(model, epsilon, layer_ids)
        ep_rt.append(epsilon)
        ls_rt.append(loss)

    return ep_rt, ls_rt


def bp_update(model, x, y, cri, opt, layer_ids, update=False):
    """
    model: 模型
    x: 输入
    y: 标签
    cri: 损失函数
    opt: 优化器
    """
    t = model(x)
    opt.zero_grad()
    loss = cri(t, y)
    # print(loss)
    loss.backward()
    grad = []
    if update:
        opt.step()
    grad = []
    # i=0
    for param in model.parameters():
        # if i in layer_ids:
        grad.append(param.grad)
    
        # i+=1
    # print(i)
    # exit()
    return grad

def grad_diff(es_grad, bp_grad):
    """
    返回各层梯度估计值与真实值的均值与方差
    """
    global cnttt
    cnttt += 1
    mean = []
    var = []
    for i in range(len(es_grad)):
        diff = es_grad[i] - bp_grad[i]
        # ratio = bp_grad[i] / es_grad[i]
        # ratio = ratio.reshape(-1)
        
        # fig, ax = plt.subplots()
        # # ax2 = ax.twinx()
        # # mask = (ratio <= (ratio.mean().item() * 100))
        # plt.title("count: (<=50): {}  (50, 150): {}  [150, 200]: {}  (>200): {}".format(torch.sum(ratio<=50).item(), torch.sum(((ratio>50) & (ratio<150))).item(), torch.sum(((ratio>=150) & (ratio<=200))).item(), torch.sum(ratio>200).item()))
        # thres = torch.ones_like(ratio).cpu() * min(200, ratio.max().item() * 1.1)
        # thres2 = torch.ones_like(ratio).cpu() * max(-200, ratio.min().item() * 1.1)
        # ratio = torch.clamp(ratio, -200, 200).cpu()
        # ax.plot(range(len(ratio)), ratio, "r")
        # ax.plot(range(len(ratio)), thres, "--")
        # ax.plot(range(len(ratio)), thres2, "--")
        # # ax2.plot(range(len(ratio)), es_grad[i].reshape(-1).cpu(), label="es_grad")
        # # ax2.plot(range(len(ratio)), bp_grad[i].reshape(-1).cpu(), label="bp_grad")
        # # plt.legend()
        # ax.set_xlabel("#param in one layer  mean(abs(es_grad)) = {:.2e} mean(abs(bp_grad)) = {:.2e}".format(es_grad[i].abs().mean(), bp_grad[i].abs().mean()))
        # ax.set_ylabel("ratio(bp_grad / es_grad)")
        # # ax2.set_ylabel("grad")
        # fig.savefig("./ratio"+'/update_{}_layer_{}_raito.jpg'.format(cnttt, i))
        # plt.cla()
        # plt.close("all")
        mean.append(torch.mean(torch.abs(diff)))
        var.append(torch.var(torch.abs(diff)))
    return [mean, var]

def update_mean_var(epoch_mv, batch_mv):
    gm,gv = epoch_mv
    bm,bv = batch_mv
    if len(gm) == 0:
        return bm,bv
    for i in range(len(gm)):
        gm[i] += bm[i]
        gv[i] += bv[i]
        # print(gm)
    return [gm, gv]


def es_train(num_epoch, model):
    lr = 0.5
    lr0 = lr
    step_size = 3
    gamma = 0.5
    co_matrices = None
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    # print(model.parameters())
    num_layers = len(model.state_dict())
    for p in model.parameters():
        print(p)

    train_acc = []
    test_acc = []
    best_acc = 0
    global_mean = []
    global_var = []
    original_time = time.asctime(time.localtime(time.time()))

    model.train()
    model = model.to(device)
    model.share_memory()
    early_break = False

    es_mode = 2
    num_directions = 40
    num_directions0 = num_directions

    backup_root = "./backups/"
    os.mkdir(backup_root + original_time)
    with open("logs.txt", "a") as fl:
        fl.write("\n\nnew case:\n")
        fl.write("original time: {}\n".format(original_time))
        fl.write("best_acc: {}\t lr0: {}\t step_size: {} \t gamma: {}\n".format(best_acc, lr0, step_size, gamma))
        fl.write("num_directions: {}\t ".format(num_directions))

    # mp.set_start_method('spawn')
    manager = Manager()
    if_alternate = False
    fall_cnt = 0
    if_es = True
    if_bp = False
    if_mirror = True
    # result = manager.list()
    start_time = time.time()

    try:
        # for layer_id in range(num_layers)[::-1]:
        #     layer_ids = [layer_id]
        # while True:
        if True:
            # for layer_id in range(num_layers):
            for ___ in range(1):
                layer_ids = list(range(num_layers))
                # layer_ids = [layer_id]
                num_directions = num_directions0
                if if_mirror:
                    num_directions = num_directions // 2
                lr = lr0
                # layer_ids = "alternate"
                for epoch in range(num_epoch):
                    print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
                    print("|", end="")
                    train_loss = 0
                    correct = 0
                    total = 0            
                    epoch_mean = []
                    epoch_var = []
                    # layer_id = 0
                    for batch_idx, (inputs, targets) in enumerate(trainloader):
                        # print("====================== layer ", layer_ids, " ======================")
                        if if_alternate:
                            layer_ids = [layer_id]
                            layer_id = (layer_id + 1) % num_layers
                        # pool = Pool(num_directions)
                        total += len(inputs)
                        ls = []
                        epsilons = []
                        processes = []
                        result = []

                        inputs, targets = inputs.to(device), targets.to(device)
                        inputs = Variable(inputs)
                        inputs.requires_grad = True
                        # torch.save(model, backup_root + original_time +"/model")
                        # torch.save(inputs, backup_root + original_time +"/inputs")
                        # torch.save(targets, backup_root + original_time +"/targets")
                        # torch.save(criterion, backup_root + original_time +"/criterion")
                        for _ in range(num_directions):
                            # for multi-process
                            # p = mp.Process(target=explore_one_direction, args=(model, inputs, targets, criterion, layer_ids, backup_root+original_time, result))
                            # p.start()
                            # processes.append(p)

                            # for single-process
                            epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, backup_root+original_time, result, if_mirror)
                            epsilons.extend(epsilon)
                            ls.extend(loss)
                            for l in loss:
                                train_loss += l

                        # for p in processes:
                        #     p.join()

                        # for eps_and_loss in result:
                        #     epsilon, loss = eps_and_loss
                        #     epsilons.append(epsilon)
                        #     ls.append(loss)
                        #     train_loss += loss
                        # result[:] = []

                        # before, after = list(), list()
                        # for i, param in enumerate(model.parameters()):
                        #     # print("*************************** layer {} ***************************".format(i))
                        #     # print(param)
                        #     before.append(param.clone())
                        es_grad = es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)
                        # for g in es_grad:
                        #     print(g.norm())
                        # for i, param in enumerate(model.parameters()):
                        #     # print("*************************** layer {} ***************************".format(i))
                        #     # print(param)
                        #     # after.append(param)
                        #     # print("delta {}: {}".format(i, torch.square(before[i] - param).sum()))
                        #     if i not in layer_ids:
                        #         assert torch.square(before[i] - param).sum() == 0.0, "update other layers"
                        bp_grad = bp_update(model, inputs, targets, criterion, optimizer, layer_ids, update=if_bp)
                        [batch_mean, batch_var] = grad_diff(es_grad, bp_grad)
                        # print(batch_mean)
                        [epoch_mean, epoch_var] = update_mean_var([epoch_mean, epoch_var], [batch_mean, batch_var])
                        outputs = model(inputs)
                        _, predicted = outputs.max(1)
                        # total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()
                        if batch_idx % 10 == 0:
                            print('=', end='')
                    print('|', 'Accuracy:', 100. * correct / total, '% ', correct, '/', total)
                    best_acc = test(test_acc, best_acc, model)
                    ######################################################
                    for i in range(len(epoch_mean)):
                        epoch_mean[i] /= total
                        epoch_var[i] /= total

                    global_mean.append(epoch_mean)
                    global_var.append(epoch_var)
                    print(epoch_mean)
                    print(epoch_var)
                    ######################################################
                    if epoch % step_size == 0 and epoch:
                        lr *= gamma
                        lr = max(lr, 0.0125)
                        if epoch % (step_size * 2) == 0: 
                            num_directions = max(int(num_directions/gamma), num_directions + 1)
                        pass

                    train_acc.append(correct / total)
                    if train_acc[-1] >= 0.88 and epoch % step_size == 0:
                        # if_es = False
                        # if_bp = True
                        # lr = 0.001
                        # lr *= 0.1
                        # print("change to bp")
                        # if_alternate = True
                        # es_mode = 2
                        # lr = 1
                        # lr *=  gamma
                        pass
                    if epoch >= 2:
                        if train_acc[-1] - train_acc[-2] < 0.01 and train_acc[-2] - train_acc[-3] < 0.01:
                            fall_cnt += 1
                            # if fall_cnt == 2:
                            #     layer_ids.pop(-1)
                            #     # print("drop last layer")
                            #     fall_cnt = 0
                            #     if len(layer_ids) == 0:
                            #         break
                        else:
                            fall_cnt = 0
                                # lr = lr0
                                # num_directions = num_directions0
                    print('Current learning rate: ', lr)
                    print('Current num_directions: ', num_directions, "mirror" if if_mirror else "")
                    now_time = time.time()
                    print("used: {}  est: {}".format(now_time - start_time, (now_time - start_time) / (epoch + 1) * (num_epoch - epoch - 1)))
    except KeyboardInterrupt:
        early_break = True
        with open("logs.txt", "a") as fl:
            fl.write("\n\nnew case:\n")
            fl.write("original time: {}\n".format(original_time))
            finish_time = time.asctime(time.localtime(time.time()))
            fl.write("finish time: {}\n".format(finish_time))
            fl.write("best_acc: {}\t lr0: {}\t lr: {} \t step_size: {} \t gamma: {}\n".format(best_acc, lr0, lr, step_size, gamma))
            fl.write("num_directions0: {} \t num_directions: {}\n ".format(num_directions0, num_directions))
            fl.write("train_accuracy {}\n".format(np.array(train_acc)))
            fl.write("test_accuracy {}".format(np.array(test_acc)))
        torch.save(model, "./models/" + str(best_acc) + finish_time + "_" + str(lr0) + "_" + str(step_size) + "_" + str(gamma))
    if not early_break:
        with open("logs.txt", "a") as fl:
            fl.write("\n\nnew case:\n")
            fl.write("original time: {}\n".format(original_time))
            finish_time = time.asctime(time.localtime(time.time()))
            fl.write("finish time: {}\n".format(finish_time))
            fl.write("best_acc: {}\t lr0: {}\t lr: {}\t step_size: {} \t gamma: {}\n".format(best_acc, lr0, lr, step_size, gamma))
            fl.write("num_directions0: {} \t num_directions: {}\n ".format(num_directions0, num_directions))
            fl.write("train_accuracy {}\n".format(np.array(train_acc)))
            fl.write("test_accuracy {}".format(np.array(test_acc)))
        torch.save(model, "./models/" + str(best_acc) + finish_time + "_" + str(lr0) + "_" + str(step_size) + "_" + str(gamma))

    print('Best training accuracy overall: ', best_acc)
    
    #########################
    # draw
    # num_layers = len(global_mean[0])
    assert num_layers == len(global_mean[0]), "num_layers error"
    mean_by_layers = [[] for i in range(num_layers)]
    var_by_layers = [[] for i in range(num_layers)]
    for epoch in range(num_epoch):
        for layer in range(num_layers):
            mean_by_layers[layer].append(global_mean[epoch][layer].cpu().numpy())
            var_by_layers[layer].append(global_var[epoch][layer].cpu().numpy())
    
    layers_range = [i for i in range(num_epoch)]
    if not os.path.exists(backup_root+original_time+'/grad/'):
        os.mkdir(backup_root+original_time+'/grad/')
    fig, ax = plt.subplots()
    ax.plot(range(num_epoch), train_acc)
    fig.savefig(backup_root+original_time+'/train_acc.jpg')
    fig, ax = plt.subplots()
    ax.plot(range(num_epoch), test_acc)
    fig.savefig(backup_root+original_time+'/test_acc.jpg')
    for layer in range(num_layers):
        fig, ax = plt.subplots()
        ax.plot(layers_range, mean_by_layers[layer])
        fig.savefig(backup_root+original_time+'/grad/layer_{}_mean.jpg'.format(layer))
        fig, ax = plt.subplots()
        ax.plot(layers_range, var_by_layers[layer])
        fig.savefig(backup_root+original_time+'/grad/layer_{}_var.jpg'.format(layer))
    ######################################################
    return train_acc, test_acc, best_acc

#####################################################################################################


def cma_es_train(num_epoch, model):
    alpha_miu = 0.01
    lr0 = alpha_miu
    lr = lr0
    alpha_sigma = 0.01
    alpha_cp = 0.01
    alpha_c1 = 0.01
    alpha_clambda = 0.01
    damping_factor = 0.01

    num_layers = len(model.state_dict())
    co_matrices = []
    ps_sigma = []
    ps_c = []
    sigmas = []
    for p in model.parameters():
        print(p)
        sz = p.cpu().detach().numpy().size
        co_matrices.append(torch.eye(sz))
        ps_sigma.append(0)
        ps_c.append(0)
        sigmas.append(0.01)

    step_size = 10
    gamma = 0.5
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    # print(model.parameters())

    train_acc = []
    test_acc = []
    best_acc = 0
    global_mean = []
    global_var = []
    original_time = time.asctime(time.localtime(time.time()))

    model.train()
    model = model.to(device)
    model.share_memory()
    early_break = False

    es_mode = 2
    num_directions = 100
    num_directions0 = num_directions
    elite_rate = 0.4
    elite_num = max(int(num_directions * elite_rate), 1)

    backup_root = "./backups/"
    os.mkdir(backup_root + original_time)
    with open("logs.txt", "a") as fl:
        fl.write("\n\nnew case:\n")
        fl.write("original time: {}\n".format(original_time))
        fl.write("best_acc: {}\t lr0: {}\t step_size: {} \t gamma: {}\n".format(best_acc, lr0, step_size, gamma))
        fl.write("num_directions: {}\t ".format(num_directions))

    # mp.set_start_method('spawn')
    manager = Manager()
    if_alternate = False
    fall_cnt = 0
    if_es = True
    if_bp = False
    # result = manager.list()
    start_time = time.time()
    try:
        # for layer_id in range(num_layers)[::-1]:
        #     layer_ids = [layer_id]
        for __ in range(1):
            layer_ids = list(range(num_layers))
            # layer_ids = "alternate"
            for epoch in range(num_epoch):
                elite_num = max(int(num_directions * elite_rate), 1)
                print("\nES layer ", "alternate" if if_alternate else layer_ids, "  Epoch: {}".format(epoch))
                print("|", end="")
                train_loss = 0
                correct = 0
                total = 0            
                epoch_mean = []
                epoch_var = []
                layer_id = 0
                for batch_idx, (inputs, targets) in enumerate(trainloader):
                    # print("====================== layer ", layer_ids, " ======================")
                    if if_alternate:
                        layer_ids = [layer_id]
                        layer_id = (layer_id + 1) % num_layers
                    # pool = Pool(num_directions)
                    total += len(inputs)
                    ls = []
                    epsilons = []
                    processes = []
                    result = []

                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = Variable(inputs)
                    inputs.requires_grad = True
                    # torch.save(model, backup_root + original_time +"/model")
                    # torch.save(inputs, backup_root + original_time +"/inputs")
                    # torch.save(targets, backup_root + original_time +"/targets")
                    # torch.save(criterion, backup_root + original_time +"/criterion")
                    for _ in range(num_directions):
                        # for multi-process
                        # p = mp.Process(target=explore_one_direction, args=(model, inputs, targets, criterion, layer_ids, backup_root+original_time, result))
                        # p.start()
                        # processes.append(p)

                        # for single-process
                        epsilon, loss = explore_one_direction(model, inputs, targets, criterion, layer_ids, co_matrices, backup_root+original_time, result)
                        epsilons.append(epsilon)
                        ls.append(loss)
                        train_loss += loss

                    # for p in processes:
                    #     p.join()

                    # for eps_and_loss in result:
                    #     epsilon, loss = eps_and_loss
                    #     epsilons.append(epsilon)
                    #     ls.append(loss)
                    #     train_loss += loss
                    # result[:] = []

                    # before, after = list(), list()
                    # for i, param in enumerate(model.parameters()):
                    #     # print("*************************** layer {} ***************************".format(i))
                    #     # print(param)
                    #     before.append(param.clone())
                    params = [alpha_miu, alpha_sigma, alpha_cp, alpha_c1, alpha_clambda,
                              damping_factor, ps_sigma, co_matrices, sigmas, ps_c]
                    es_grad, params = cma_es_update(model, epsilons, ls, lr, layer_ids, es_mode, update=if_es)

                    # alpha_miu, alpha_sigma, alpha_cp, alpha_c1, alpha_clambda = params[:5]
                    # damping_factor = params[5]
                    # ps_sigma = params[6]
                    # co_matrices = params[7]
                    # sigmas = params[8]
                    # ps_c = sigmas[9]
                    ps_sigma, sigmas, ps_c, co_matrices = params
                    # for i, param in enumerate(model.parameters()):
                    #     # print("*************************** layer {} ***************************".format(i))
                    #     # print(param)
                    #     # after.append(param)
                    #     # print("delta {}: {}".format(i, torch.square(before[i] - param).sum()))
                    #     if i not in layer_ids:
                    #         assert torch.square(before[i] - param).sum() == 0.0, "update other layers"
                    bp_grad = bp_update(model, inputs, targets, criterion, optimizer, layer_ids, update=if_bp)
                    [batch_mean, batch_var] = grad_diff(es_grad, bp_grad)
                    # print(batch_mean)
                    [epoch_mean, epoch_var] = update_mean_var([epoch_mean, epoch_var], [batch_mean, batch_var])
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    # total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    if batch_idx % 10 == 0:
                        print('=', end='')
                print('|', 'Accuracy:', 100. * correct / total, '% ', correct, '/', total)
                best_acc = test(test_acc, best_acc, model)
                ######################################################
                for i in range(len(epoch_mean)):
                    epoch_mean[i] /= total
                    epoch_var[i] /= total
                global_mean.append(epoch_mean)
                global_var.append(epoch_var)
                print(epoch_mean)
                print(epoch_var)
                ######################################################
                if epoch % step_size == 0 and epoch:
                    # lr *= gamma
                    # num_directions = max(int(num_directions/gamma), num_directions + 1)
                    pass

                train_acc.append(correct / total)
                if train_acc[-1] >= 0.88 and epoch % step_size == 0:
                    # if_es = False
                    # if_bp = True
                    # lr = 0.001
                    # lr *= 0.1
                    # print("change to bp")
                    # if_alternate = True
                    # es_mode = 2
                    # lr = 1
                    # lr *=  gamma
                    pass
                if epoch >= 2:
                    if train_acc[-1] - train_acc[-2] < 0.01 and train_acc[-2] - train_acc[-3] < 0.01:
                        fall_cnt += 1
                        # if fall_cnt == 2:
                        #     layer_ids.pop(-1)
                        #     # print("drop last layer")
                        #     fall_cnt = 0
                        #     if len(layer_ids) == 0:
                        #         break
                    else:
                        fall_cnt = 0
                            # lr = lr0
                            # num_directions = num_directions0
                print('Current learning rate: ', lr)
                print('Current num_directions: ', num_directions)
                now_time = time.time()
                print("used: {} est: {}".format(now_time - start_time, (now_time - start_time) / (epoch + 1) * (num_epoch - epoch - 1)))
    except KeyboardInterrupt:
        early_break = True
        with open("logs.txt", "a") as fl:
            fl.write("\n\nnew case:\n")
            fl.write("original time: {}\n".format(original_time))
            finish_time = time.asctime(time.localtime(time.time()))
            fl.write("finish time: {}\n".format(finish_time))
            fl.write("best_acc: {}\t lr0: {}\t lr: {} \t step_size: {} \t gamma: {}\n".format(best_acc, lr0, lr, step_size, gamma))
            fl.write("num_directions0: {} \t num_directions: {}\n ".format(num_directions0, num_directions))
            fl.write("train_accuracy {}\n".format(np.array(train_acc)))
            fl.write("test_accuracy {}".format(np.array(test_acc)))
        torch.save(model, "./models/" + str(best_acc) + finish_time + "_" + str(lr0) + "_" + str(step_size) + "_" + str(gamma))
    if not early_break:
        with open("logs.txt", "a") as fl:
            fl.write("\n\nnew case:\n")
            fl.write("original time: {}\n".format(original_time))
            finish_time = time.asctime(time.localtime(time.time()))
            fl.write("finish time: {}\n".format(finish_time))
            fl.write("best_acc: {}\t lr0: {}\t lr: {}\t step_size: {} \t gamma: {}\n".format(best_acc, lr0, lr, step_size, gamma))
            fl.write("num_directions0: {} \t num_directions: {}\n ".format(num_directions0, num_directions))
            fl.write("train_accuracy {}\n".format(np.array(train_acc)))
            fl.write("test_accuracy {}".format(np.array(test_acc)))
        torch.save(model, "./models/" + str(best_acc) + finish_time + "_" + str(lr0) + "_" + str(step_size) + "_" + str(gamma))

    print('Best training accuracy overall: ', best_acc)
    
    #########################
    # draw
    # num_layers = len(global_mean[0])
    assert num_layers == len(global_mean[0]), "num_layers error"
    mean_by_layers = [[] for i in range(num_layers)]
    var_by_layers = [[] for i in range(num_layers)]
    for epoch in range(num_epoch):
        for layer in range(num_layers):
            mean_by_layers[layer].append(global_mean[epoch][layer].cpu().numpy())
            var_by_layers[layer].append(global_var[epoch][layer].cpu().numpy())
    
    layers_range = [i for i in range(num_epoch)]
    if not os.path.exists(backup_root+original_time+'/grad/'):
        os.mkdir(backup_root+original_time+'/grad/')
    for layer in range(num_layers):
        fig, ax = plt.subplots()
        ax.plot(layers_range, mean_by_layers[layer])
        fig.savefig(backup_root+original_time+'/grad/layer_{}_mean.jpg'.format(layer))
        fig, ax = plt.subplots()
        ax.plot(layers_range, var_by_layers[layer])
        fig.savefig(backup_root+original_time+'/grad/layer_{}_var.jpg'.format(layer))
    ######################################################
    return train_acc, test_acc, best_acc


def bp_train(num_epoch, model):
    lr = 0.005
    lr0 = lr
    step_size = 100
    gamma = 0.5
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    train_acc = []
    test_acc = []
    best_acc = 0

    model = model.to(device)

    for epoch in range(num_epoch):
        # model.train()
        print('\nEpoch: ', epoch)
        print("|", end="")
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if batch_idx % 10 == 0:
                print('=', end='')

        print('|', 'Accuracy:', 100. * correct / total,'% ', correct, '/', total)
        best_acc = test(test_acc, best_acc, model)

        train_acc.append(correct / total)
        scheduler.step()
        print('Current learning rate: ', scheduler.get_last_lr()[0])

    print('Best training accuracy overall: ', best_acc)
    return train_acc, test_acc, best_acc




def w_test():
    w_net = LinearNet()
    # bp_train(30, w_net)
    es_train(5, w_net)
    # cma_es_train(100, w_net)

if __name__ == "__main__":
    # decomposed_net = build(decomp=True)
    # es_train(10, decomposed_net)
    full_net = build(decomp=False)
    # print(full_net)
    # for param in full_net.parameters():
    #     print(param.shape)
    # bp_train(30, full_net)
    es_train(40, full_net)
    # w_test()

    # for param_tensor in decomposed_net.state_dict(): # 字典的遍历默认是遍历 key，所以param_tensor实际上是键值
    #     print(param_tensor,'\t',decomposed_net.state_dict()[param_tensor].size())
    # layer_num = 0
    # for param in decomposed_net.parameters():
    #     layer_num += 1
    # print(layer_num)
    # exit(0)
