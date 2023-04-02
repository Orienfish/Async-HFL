import logging
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
import time
from sklearn.decomposition import PCA

class Client(object):
    """Simulated federated learning client."""

    def __init__(self, client_id):
        self.client_id = client_id
        self.available = True  # When the client is running local training,
                               # it is not available
                               # This property is controlled by the server
        self.loss = None
        self.delay = None
        self.pca = None

    def __repr__(self):
        #return 'Client #{}: {} samples in labels: {}'.format(
        #    self.client_id, len(self.data), set([label for _, label in self.data]))
        return 'Client #{}'.format(self.client_id)

    # Set non-IID data configurations
    def set_bias(self, pref, bias):
        self.pref = pref
        self.bias = bias

    def set_shard(self, shard):
        self.shard = shard

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return argv.copy()
        except:
            return argv

    def upload(self, argv):
        # Upload to the server
        try:
            return argv.copy()
        except:
            return argv

    def set_available(self):
        self.available = True

    def set_unavailable(self):
        self.available = False

    # Federated learning phases
    def set_data(self, data, config):
        # Extract from config
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition
        self.dataset = config.model

        # Download data
        self.data = self.download(data)

        # Extract trainset, testset (if applicable)
        data = self.data
        if do_test:  # Partition for testset if applicable
            self.trainset = data[:int(len(data) * (1 - test_partition))]
            self.testset = data[int(len(data) * (1 - test_partition)):]
        else:
            self.trainset = data

        # Set num samples
        self.num_samples = len(data)

    # Federated learning phases
    def set_data_leaf(self, train_data, test_data, config):
        # Extract from config
        do_test = self.do_test = config.clients.do_test
        test_partition = self.test_partition = config.clients.test_partition
        self.dataset = config.model

        # Download data
        self.data = self.download(train_data)

        # Extract trainset, testset (if applicable)
        if do_test:  # Partition for testset if applicable
            self.trainset = train_data
            self.testset = test_data
        else:
            self.trainset = train_data

        # Set num samples
        self.num_samples = len(train_data['x'])

    def set_gateway(self, gateway_id):
        self.gateway_id = gateway_id

    def set_delay_uniform(self):
        # Set the link speed and delay for the upcoming run
        # Set the minimum value to avoid negative values
        link_speed = max(random.normalvariate(self.speed_mean, self.speed_std), 10.0)
        comp_time = max(random.normalvariate(self.comp_mean, self.comp_std), 1.0)
        self.delay = (self.model_size / link_speed) + comp_time  # upload delay in sec
        logging.debug('client {} link speed: {} comp time: {} delay: {}'.format(
            self.client_id, link_speed, comp_time, self.delay
        ))
        # self.throughput = self.model_size / self.delay

    def set_delay_to_gateway(self, comm_delay, comp_delay, model_size,
                             speed_mean=None, speed_std=None, comp_std=None):
        # Set model size
        self.model_size = model_size

        # Set mean speed in the uniform delay mode
        self.speed_mean = speed_mean
        self.speed_std = speed_std

        # Set mean comp time
        self.comp_mean = comp_delay
        self.comp_std = comp_std

        # Set estimated delay
        self.delay = self.est_delay = comm_delay + comp_delay
        self.throughput = self.model_size / self.delay

    def sync_global_configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent global model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Client {} load global model: {}'.format(
            self.client_id, path))

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

    def async_global_configure(self, config, download_time):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent global model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Client {} load global model: {}'.format(
            self.client_id, path))

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)


    def sync_client_configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent gateway model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/gateway{}'.format(self.gateway_id)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.debug('Client {} load gateway {} model: {}'.format(
            self.client_id, self.gateway_id, path))

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

        # Set the delay for the upcoming run if delay_mode is uniform
        if config.delay_mode == 'uniform':
            self.set_delay_uniform()


    def async_client_configure(self, config, gateway_download_time):
        import fl_model  # pylint: disable=import-error

        # Extract from config
        self.model_path = config.paths.model

        # Download from server
        config = self.download(config)

        # Extract machine learning task from config
        self.task = config.fl.task
        self.epochs = config.fl.epochs
        self.batch_size = config.fl.batch_size

        # Download most recent gateway model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/gateway{}_{}'.format(self.gateway_id,
                                                         gateway_download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.debug('Client {} load gateway {} model: {}'.format(
            self.client_id, self.gateway_id, path))

        # Create optimizer
        self.optimizer = fl_model.get_optimizer(self.model)

        # Set the delay for the upcoming run if delay_mode is uniform
        if config.delay_mode == 'uniform':
            self.set_delay_uniform()

    def run(self, reg=None, rou=None):
        # Perform federated learning task
        {
            "train": self.train(reg=reg, rou=rou)
        }[self.task]

    def get_report(self):
        # Report results to server.
        return self.upload(self.report)

    # Machine learning tasks
    def train(self, reg=None, rou=None):
        import fl_model  # pylint: disable=import-error

        old_weights = fl_model.extract_weights(self.model)

        # Perform model training
        trainloader = fl_model.get_trainloader(self.trainset, self.batch_size)
        self.loss = fl_model.train(self.model, trainloader,
                                   self.optimizer, self.epochs, reg, rou)

        #logging.info('Training on client #{}, loss {} mean delay {}s'.format(
        #    self.client_id, self.loss, self.delay))

        # Extract model weights and biases
        self.weights = fl_model.extract_weights(self.model)
        if self.dataset == 'Shakespeare' or self.dataset == 'HPWREN':
            # LSTM suffers from vanishing gradients
            self.grads = self.extract_delta_weights(self.weights, old_weights)
        else:
            self.grads = fl_model.extract_grads(self.model)

        # Transform the gradients data using PCA if requested
        if self.pca is not None:
            self.grads = self.flatten_weights(self.grads).reshape((1, -1))
            self.grads = self.pca.transform(self.grads).reshape(-1)
        else:
            self.grads = self.flatten_weights(self.grads)

        # Generate report for gateway
        self.report = Report(self, self.weights, self.grads, self.loss, self.delay)


    def test(self, model):
        # Perform local model testing given global model
        import fl_model

        testloader = fl_model.get_testloader(self.testset, self.batch_size)
        test_loss, accuracy = fl_model.test(model, testloader)

        self.report.test_loss = test_loss
        self.report.accuracy = accuracy


    def extract_delta_weights(self, new_weights, old_weights):
        # Extract the delta between new weights and old weights
        deltas = []
        for i, (name, w) in enumerate(new_weights):
            bl_name, baseline = old_weights[i]

            # Ensure correct weight is being updated
            assert name == bl_name

            # Calculate update
            delta = w - baseline
            deltas.append((name, delta))

        return deltas

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)


class Report(object):
    """Federated learning client report."""
    def __init__(self, client, weights, grads, loss, delay):
        self.client_id = client.client_id
        self.num_samples = client.num_samples
        self.weights = weights
        self.grads = grads
        self.loss = loss
        self.delay = delay
