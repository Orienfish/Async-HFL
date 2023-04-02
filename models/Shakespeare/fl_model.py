import load_data
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import numpy as np
import time
import json
import os
from utils.language_utils import letter_to_index
import copy

SEQ_LEN = 80
NUM_CLASSES = 80
NUM_HIDDEN = 100

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
        self.trainset = {'users': [], 'user_data': {}, 'num_samples': []}
        self.labels = []
        trainset_size = 0

        train_dir = os.path.join(path, 'train')

        for file in os.listdir(train_dir):
            with open(os.path.join(train_dir, file)) as json_file:
                logging.info('loading {}'.format(os.path.join(train_dir, file)))
                data = json.load(json_file)
                self.trainset['users'] += data['users']
                self.trainset['num_samples'] += data['num_samples']
                for user in data['users']:
                    self.labels += data['user_data'][user]['y']
                    self.labels = list(set(self.labels))

                    trainset_size += len(data['user_data'][user]['y'])

                    # Convert letters to ints in user data
                    self.trainset['user_data'][user] = {}
                    self.trainset['user_data'][user]['x'] = []
                    for x_sample in data['user_data'][user]['x']:
                        self.trainset['user_data'][user]['x'].append(
                            self.process_x(x_sample)
                        )
                    self.trainset['user_data'][user]['y'] = []
                    for y_sample in data['user_data'][user]['y']:
                        self.trainset['user_data'][user]['y'].append(
                            self.process_y(y_sample)
                        )

        self.labels.sort()
        print(len(self.labels))
        self.trainset_size = trainset_size

        self.testset = {'users': [], 'user_data': {}, 'num_samples': []}
        test_dir = os.path.join(path, 'test')

        for file in os.listdir(test_dir):
            with open(os.path.join(test_dir, file)) as json_file:
                logging.info('loading {}'.format(os.path.join(test_dir, file)))
                data = json.load(json_file)
                self.testset['users'] += data['users']
                self.testset['num_samples'] += data['num_samples']

                for user in data['users']:
                    # Convert letters to ints in user data
                    self.testset['user_data'][user] = {}
                    self.testset['user_data'][user]['x'] = []
                    for x_sample in data['user_data'][user]['x']:
                        self.testset['user_data'][user]['x'].append(
                            self.process_x(x_sample)
                        )
                    self.testset['user_data'][user]['y'] = []
                    for y_sample in data['user_data'][user]['y']:
                        self.testset['user_data'][user]['y'].append(
                            self.process_y(y_sample)
                        )


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
        self.seq_len = SEQ_LEN  # window_size?
        self.num_classes = NUM_CLASSES  # vocab_size?
        self.embedding_dim = 8
        self.n_hidden = NUM_HIDDEN
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x, prev_state):  # x: (n_samples, seq_len)
        emb = self.embedding(x)  # (n_samples, seq_len, embedding_dim)
        x, state = self.lstm(emb, prev_state)  # (n_samples, seq_len, n_hidden)
        logits = self.fc(x[:, -1, :])
        return logits, state

    def zero_state(self, batch_size):
        return (Variable(torch.zeros(1, batch_size, self.n_hidden)),
                Variable(torch.zeros(1, batch_size, self.n_hidden)))


def get_optimizer(model):
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):
    # Convert the dictionary-format of trainset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_train = np.array(trainset['x'], dtype=np.float32)
    x_train = torch.LongTensor(x_train)
    y_train = np.array(trainset['y'], dtype=np.int32)
    y_train = torch.Tensor(y_train).type(torch.int64)

    train_dataset = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_loader


def get_testloader(testset, batch_size):
    # Convert the dictionary-format of testset (with keys of 'x' and 'y') to
    # TensorDataset, then create the DataLoader from it
    x_test = np.array(testset['x'], dtype=np.float32)
    x_test = torch.LongTensor(x_test)
    y_test = np.array(testset['y'], dtype=np.int32)
    y_test = torch.Tensor(y_test).type(torch.int64)

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


def train(model, trainloader, optimizer, epochs, reg=True, rou=None):
    old_model = copy.deepcopy(model)
    old_model.to(device)
    old_model.eval()

    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)

    batch_size, state_h, state_c = None, None, None

    for epoch in range(1, epochs + 1):
        train_loss, train_gw_l2_loss = 0, 0
        correct = 0
        for batch_id, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            if batch_size is None:
                batch_size = image.shape[0]
                state_h, state_c = model.zero_state(batch_size)
                state_h = state_h.to(device)
                state_c = state_c.to(device)

            if image.shape[0] < batch_size:  # Less than one batch
                break

            output, (state_h, state_c) = model(image, (state_h, state_c))

            state_h = state_h.detach()
            state_c = state_c.detach()
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

    batch_size, state_h, state_c = None, None, None

    with torch.no_grad():
        for image, label in testloader:
            image, label = image.to(device), label.to(device)

            if batch_size is None:
                batch_size = image.shape[0]
                state_h, state_c = model.zero_state(batch_size)
                state_h = state_h.to(device)
                state_c = state_c.to(device)

            if image.shape[0] < batch_size:  # Less than one batch
                break

            output, (state_h, state_c) = model(image, (state_h, state_c))

            state_h = state_h.detach()
            state_c = state_c.detach()

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
