import logging
import pickle
import random
import torch
import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threading import Thread
from server import Server
from .record import Record, Profile
from .dqn import DQN, select_action_test

# Cuda settings
use_cuda = torch.cuda.is_available()
device = torch.device(  # pylint: disable=no-member
    'cuda' if use_cuda else 'cpu')

WEIGHT_PCA_DIM = 100

class DQNServer(Server):
    """DQN federated learning server."""

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Load DQN model for client selection
        total_clients = self.config.clients.total
        dqn_model_path = self.config.paths.model + '/dqn'
        logging.info('Loading trained DQN model from {}...'.format(dqn_model_path))
        self.dqn_net = DQN(n_input=WEIGHT_PCA_DIM * (total_clients + 1),
                           n_output=total_clients).to(device=device, dtype=torch.float32)
        self.dqn_net.load_state_dict(torch.load(dqn_model_path))
        self.dqn_net.eval()

        # Init self accuracy records
        self.records = Record("t", "acc", "throughput")

    def make_clients(self, num_clients):
        super().make_clients(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            client.set_link(self.config)
            speed.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients, self.loader.labels)
        if not self.config.data.IID:
            self.profile.set_primary_label([client.pref for client in self.clients])

    def make_clients_leaf(self, num_clients):
        super().make_clients_leaf(num_clients)

        # Set link speed for clients
        speed = []
        for client in self.clients:
            client.set_link(self.config)
            speed.append(client.speed_mean)

        logging.info('Speed distribution: {} Kbps'.format([s for s in speed]))

        # Initiate client profile of loss and delay
        self.profile = Profile(num_clients, self.loader.labels)
        if not self.config.data.IID:
            self.profile.set_primary_label(
                [client.pref for client in self.clients])

    # Init the environment for one episode of training/testing
    def episode_init(self):
        import fl_model

        # Set up simulated server
        self.load_data()
        self.load_model()
        if self.config.loader != 'leaf':
            self.num_clients = self.config.clients.total
            self.make_clients(self.num_clients)
        else:
            self.num_clients = min(self.config.clients.total, self.loader.num_clients)
            self.make_clients_leaf(self.num_clients)

        # Init the server and state for the current episode
        _ = [client.set_delay() for client in self.clients]
        self.configuration(self.clients)
        threads = [Thread(target=client.run) for client in self.clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        reports = self.reporting(self.clients)
        # Update profile and plot
        self.update_profile(reports)
        self.profile.plot(0.0, self.config.paths.plot)

        # Perform weight aggregation
        updated_weights = self.aggregation(reports)

        # Load updated weights to global model and save
        fl_model.load_weights(self.model, updated_weights)
        self.save_model(self.model, self.config.paths.model)

        # Train PCA and gather initial state (weights after PCA)
        self.weights_array = [self.flatten_weights(updated_weights)]
        for report in reports:
            self.weights_array.append(self.flatten_weights(report.weights))
        self.weights_array = np.array(self.weights_array)

        # Scaling the weights collected from all clients
        self.scaler = StandardScaler()
        self.scaler.fit(self.weights_array)
        self.weights_array_scale = self.scaler.transform(self.weights_array)

        # PCA transform
        self.pca = PCA(n_components=WEIGHT_PCA_DIM)
        self.pca.fit(self.weights_array_scale)

        state = torch.tensor(self.pca.transform(self.weights_array_scale),
                             device=device, dtype=torch.float32).view(1, -1)
        return state

    # Run synchronous federated learning
    def run(self):
        rounds = self.config.fl.rounds
        target_accuracy = self.config.fl.target_accuracy
        reports_path = self.config.paths.reports

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Init environment for episode
        self.state = self.episode_init()

        # Perform rounds of federated learning
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the sync federated learning round
            accuracy, T_new = self.sync_round(round, T_old)
            logging.info('Round finished at time {} s\n'.format(T_new))

            # Update time
            T_old = T_new

            # Break loop when target accuracy is met
            if target_accuracy and (accuracy >= target_accuracy):
                logging.info('Target accuracy reached.')
                break

        if reports_path:
            with open(reports_path, 'wb') as f:
                pickle.dump(self.saved_reports, f)
            logging.info('Saved reports: {}'.format(reports_path))

    def sync_round(self, round, T_old):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        sample_clients = self.selection()
        _ = [client.set_delay() for client in sample_clients]
        self.throughput = sum([client.throughput for client in sample_clients])
        logging.info('Avg throughput {} kB/s'.format(self.throughput))

        # Configure sample clients
        self.configuration(sample_clients)

        # Use the max delay in all sample clients as the delay in sync round
        max_delay = max([c.delay for c in sample_clients])

        # Run clients using multithreading for better parallelism
        threads = [Thread(target=client.run) for client in sample_clients]
        [t.start() for t in threads]
        [t.join() for t in threads]
        T_cur = T_old + max_delay  # Update current time

        # Receive client updates
        reports = self.reporting(sample_clients)

        # Update profile and plot
        self.update_profile(reports)
        # Plot every plot_interval
        #if math.floor(T_cur / self.config.plot_interval) > \
        #        math.floor(T_old / self.config.plot_interval):
        #    self.profile.plot(T_cur, self.config.paths.plot)

        # Perform weight aggregation
        logging.info('Aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.save_reports(round, reports)

        # Save updated global model
        self.save_model(self.model, self.config.paths.model)

        # Update state
        self.weights_array[0] = self.flatten_weights(updated_weights)  # global weights
        for report in reports:
            self.weights_array[report.client_id] = self.flatten_weights(report.weights)
        self.weights_array_scale = self.scaler.transform(self.weights_array)
        self.state = torch.tensor(self.pca.transform(self.weights_array_scale),
                                  device=device, dtype=torch.float32).view(1, -1)

        # Test global model accuracy
        if self.config.clients.do_test:  # Get average accuracy from client reports
            accuracy = self.accuracy_averaging(reports)
        else:  # Test updated model on server
            if self.config.loader != 'leaf':
                testset = self.loader.get_testset()
            else:
                testset = self.loader.get_testset(self.select_loader_client)
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            accuracy = fl_model.test(self.model, testloader)

        logging.info('Average accuracy: {:.2f}%'.format(100 * accuracy))
        self.records.append_record(t=T_cur, acc=accuracy, throughput=self.throughput)
        return self.records.get_latest_acc(), self.records.get_latest_t()

    def selection(self):
        # Select devices to participate in round
        clients_per_round = self.config.clients.per_round

        # Select clients randomly
        action = select_action_test(self.state, self.dqn_net,
                                    self.num_clients, clients_per_round)
        # Convert action to numpy
        action = action.view(-1).detach().cpu().numpy()

        sample_clients = [self.clients[i] for i in action]

        logging.info('Select clients: {}'.format(sample_clients))

        return sample_clients

    def update_profile(self, reports):
        for report in reports:
            self.profile.update(report.client_id, report.loss, report.delay,
                                self.flatten_weights(report.weights),
                                self.flatten_weights(report.grads))
