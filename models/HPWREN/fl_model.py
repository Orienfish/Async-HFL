import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import copy
import json
import os
from utils.language_utils import letter_to_index

SEQ_LEN = 24
NUM_INPUT = 12  # input_size, corresponds to the number of features in the input
NUM_OUTPUT = 12  # output_size, the number of items in the output
NUM_HIDDEN = 128
NUM_LAYERS = 1

# Training settings
lr = 0.001
momentum = 0.9
log_interval = 10
loss_thres = 0.001

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')


class Generator(load_data.Generator):
    """Generator for MNIST dataset."""

    def read(self, path):
        self.trainset = {'users': [], 'user_data': {}, 'num_samples': []}
        self.labels = []
        trainset_size = 0

        train_file = os.path.join(path, 'train.json')
        with open(train_file) as json_file:
            logging.info('loading {}'.format(train_file))
            data = json.load(json_file)
            self.trainset['users'] += data['users']
            self.trainset['user_data'].update(data['user_data'])
            self.trainset['num_samples'] += data['num_samples']

            for user in data['users']:
                trainset_size += len(data['user_data'][user]['y'])

        self.trainset_size = trainset_size

        self.testset = {'users': [], 'user_data': {}, 'num_samples': []}

        test_file = os.path.join(path, 'test.json')
        with open(test_file) as json_file:
            logging.info('loading {}'.format(test_file))
            data = json.load(json_file)
            self.testset['users'] += data['users']
            self.testset['user_data'].update(data['user_data'])
            self.testset['num_samples'] += data['num_samples']


    def generate(self, path):
        self.read(path)

        return self.trainset

    def process_x(self, raw_x_sample):
        x_batch = [letter_to_index(letter) for letter in raw_x_sample]
        return x_batch

    def process_y(self, raw_y_sample):
        return letter_to_index(raw_y_sample)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.num_layers = NUM_LAYERS

        self.input_size = NUM_INPUT
        self.hidden_size = NUM_HIDDEN
        self.output_size = NUM_OUTPUT

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)

        out, (h_out, c_out) = self.lstm(x, (h_0, c_0))  # b, t, output_size

        out = self.fc(out)
        out = out[:, -1, :]

        return out


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    # Convert the dictionary-format of trainset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_train = np.array(trainset['x'], dtype=np.float32)
    x_train = torch.Tensor(x_train)
    y_train = np.array(trainset['y'], dtype=np.float32)
    y_train = torch.Tensor(y_train)

    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_testloader(testset, batch_size):
    # Convert the dictionary-format of testset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_test = np.array(testset['x'], dtype=np.float32)
    x_test = torch.Tensor(x_test)
    y_test = np.array(testset['y'], dtype=np.float32)
    y_test = torch.Tensor(y_test)

    test_dataset = TensorDataset(x_test, y_test)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return test_loader


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
    criterion = nn.MSELoss().to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_gw_l2_loss = 0, 0
        # correct = 0
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

            # _, predicted = output.max(1)
            # print(predicted)
            # correct += predicted.eq(label).sum().item()

    # total = len(trainloader.dataset)  # Total # of test samples
    train_loss = train_loss / len(trainloader)
    # accuracy = correct / total
    # logging.debug('Train accuracy: {}'.format(accuracy))

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
    criterion = nn.MSELoss().to(device)

    test_loss = 0
    total = len(testloader.dataset)
    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            # sum up batch loss
            test_loss += criterion(output, label).item()

    test_loss = test_loss / len(testloader)
    err = test_loss / total
    logging.debug('Test loss: {} MSE Err: {:.2f}%'.format(
        test_loss, err
    ))

    return test_loss, err
