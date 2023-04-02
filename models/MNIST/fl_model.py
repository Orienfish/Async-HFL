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
lr = 0.01
momentum = 0.9
log_interval = 10
loss_thres = 0.001

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for MNIST dataset."""

    # Extract MNIST data using torchvision datasets
    def read(self, path):
        self.trainset = datasets.MNIST(
            path, train=True, download=True, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.testset = datasets.MNIST(
            path, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.1307,), (0.3081,))
            ]))
        self.labels = list(self.trainset.classes)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


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
    total = len(testloader.dataset)
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
