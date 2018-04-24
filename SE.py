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
import models.resnet
from tensorboard_logger import configure, log_value
import config

use_cuda = torch.cuda.is_available()
OUTPATH = './checkpoint/checkpoint_se'
configure("runs/run-se", flush_secs=5)
BATCH, EPOCH = config.BATCH, config.EPOCH

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
model = models.se_resnet.resnet50(num_classes=100)
if use_cuda:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.1)
criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.data[0]))
            log_value('loss', loss, 50000*(epoch-1) + batch_idx)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), OUTPATH + str(epoch))

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
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)

        # sum up batch loss
        test_loss += criterion(output, target).data[0]
        # get the index of the max log-probability
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(test_loader.dataset),
                  100. * correct / len(test_loader.dataset)))

for epoch in range(1, EPOCH+1):
    train(epoch)
    test()
