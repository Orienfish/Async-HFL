import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import copy

# Training settings
lr = 0.001
momentum = 0.9
weight_decay = 5e-4
log_interval = 10
loss_thres = 1e-8

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')

class Generator(load_data.Generator):
    """Generator for CIFAR-10 dataset."""

    # Extract CIFAR-10 data using torchvision datasets
    def read(self, path):
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.trainset = datasets.CIFAR10(path, train=True, download=True, transform=train_transform)
        self.testset = datasets.CIFAR10(path, train=False, download=True, transform=valid_transform)
        self.labels = list(self.trainset.classes)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(Net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AvgPool2d(4)
        self.flat = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.flat(out)
        out = self.linear(out)
        return out


#def ResNet18(num_classes=10, config = [2, 2, 2, 2]):
#    return ResNet(BasicBlock, config, num_classes)

"""
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""

def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)


def extract_weights(model):
    weights = []
    state_dict = model.to(torch.device('cpu')).state_dict()
    for name in state_dict.keys():  # pylint: disable=no-member
        weight = state_dict[name]
        weights.append((name, weight))

    return weights


def load_weights(model, weights):
    updated_state_dict = {}
    for name, weight in weights:
        updated_state_dict[name] = weight

    model.load_state_dict(updated_state_dict, strict=False)


def flatten_weights(weights):
    # Flatten weights into vectors
    weight_vecs = []
    for _, weight in weights:
        weight_vecs.extend(weight.flatten().tolist())

    return np.array(weight_vecs)


def extract_grads(model):
    grads = []
    for name, weight in model.to(torch.device('cpu')).named_parameters():  # pylint: disable=no-member
        if weight.requires_grad:
            grads.append((name, weight.grad))

    return grads


def train(model, trainloader, optimizer, epochs, reg=None, rou=None):
    old_model = copy.deepcopy(model)
    old_model.to(device)
    old_model.eval()

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_gw_l2_loss = 0, 0
        correct = 0
        for batch_id, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)

            # Add regularization
            if reg is not None and rou is not None:
                gw_l2_loss = 0.0
                for paramA, paramB in zip(model.parameters(), old_model.parameters()):
                    gw_l2_loss += rou / 2 * \
                                  torch.sum(torch.square(paramA - paramB.detach()))
                loss += gw_l2_loss
                train_gw_l2_loss += gw_l2_loss.item()

            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            if batch_id % log_interval == 0:
                logging.debug('Epoch: [{}/{}]\tLoss: {:.6f}'.format(
                    epoch, epochs, loss.item()))

            # Stop training if model is already in good shape
            if loss.item() < loss_thres:
                return loss.item()

            _, predicted = output.max(1)
            # print(predicted)
            correct += predicted.eq(label).sum().item()

    total = len(trainloader.dataset)  # Total # of test samples
    train_loss = train_loss / len(trainloader)
    accuracy = correct / total
    logging.debug('Train accuracy: {}'.format(accuracy))

    if reg is not None and rou is not None:
        train_gw_l2_loss = train_gw_l2_loss / len(trainloader)
        logging.info(
            'loss: {} l2_loss: {}'.format(train_loss, train_gw_l2_loss))
    else:
        logging.info(
            'loss: {}'.format(train_loss))

    return train_loss


def test(model, testloader):
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    test_loss = 0
    correct = 0
    total = len(testloader.dataset)  # Total # of test samples
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up batch loss
            test_loss += criterion(output, label).item()
            # get the index of the max log-probability
            _, predicted = output.max(1)
            correct += predicted.eq(label).sum().item()

    test_loss = test_loss / len(testloader)
    accuracy = correct / total
    logging.debug('Test loss: {} Accuracy: {:.2f}%'.format(
        test_loss, 100 * accuracy
    ))

    return test_loss, accuracy