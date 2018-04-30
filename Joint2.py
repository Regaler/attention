# Licence: BSD
# Author: Ghassen Hamrouni

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import models.joint_resnet2
from tensorboard_logger import configure, log_value
import config

use_cuda = torch.cuda.is_available()
OUTPATH = './checkpoint/checkpoint_joint2'
configure("runs/run-joint2", flush_secs=5)
EPOCH = config.EPOCH
BATCH = config.BATCH

# Training dataset
train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='.', train=True, download=True,
            transform=transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
            ])), batch_size=BATCH, shuffle=True, num_workers=4)

# Test dataset
test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='.', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))
        ])), batch_size=BATCH, shuffle=True, num_workers=4)

#model = Net()
model = models.joint_resnet2.resnet50(num_classes=100)
model.load_state_dict(torch.load("./checkpoint/checkpoint_joint2200"))
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Joint: # of params: " + str(pytorch_total_params))

if use_cuda:
    model.cuda()
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    train_loss = 0
    correct = 0

    optimizer = optim.SGD(model.parameters(), lr=config.learning_rate(0.1, epoch), momentum=0.9, weight_decay=5e-4)
    #optimizer = optim.SGD(model.parameters(), lr=0.1*0.0008, momentum=0.9, weight_decay=5e-4)

    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        if batch_idx == 0:
            torch.save(data, './data.pkl')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
            log_value('loss', loss, 391*(epoch-1) + batch_idx)

        # sum up batch loss
        train_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    if epoch % 20 == 0:
        if torch.cuda.device_count() > 1:
            torch.save(model.module.state_dict(), OUTPATH + str(epoch))

    train_loss = train_loss / (len(train_loader.dataset) // BATCH)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(train_loss, correct, len(train_loader.dataset), 100. * correct / len(train_loader.dataset)))
    log_value('train_acc', 100. * correct / len(train_loader.dataset), epoch)

#
# Test performance on CIFAR100
#

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, requires_grad=False), Variable(target)
        output = model(data)

        # sum up batch loss
        test_loss += criterion(output, target).item()
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss = test_loss / (len(test_loader.dataset) // BATCH)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    log_value('test_loss', test_loss, epoch)
    log_value('test_acc', 100. * correct / len(test_loader.dataset), epoch)


for epoch in range(1, EPOCH+1):
    #train(epoch)
    test()
