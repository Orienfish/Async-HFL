import logging
import pickle
import numpy as np
from threading import Thread
import torch
from queue import PriorityQueue
import os
import sys
from .syncServer import SyncServer
from .record import Record
from .asyncEvent import asyncEvent

class SemiAsyncServer(SyncServer):
    """Asynchronous federated learning server."""

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        saved_model_path = self.config.paths.saved_model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.save_model(self.model, saved_model_path)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    # Warmup the clients queue by calling all clients
    def server_warmup(self):
        logging.info('Server warmup ...')

        # Init the server and state for the current episode
        _ = [client.sync_global_configure(self.config) for client in
             self.clients]

        if self.config.model not in ['CIFAR-10', 'FEMNIST', 'Shakespeare']:
            threads = [Thread(target=client.run) for client in self.clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:  # Use sequential execution because ResNet is too big
            _ = [client.run() for client in self.clients]

        # Receive client updates
        reports = self.reporting(self.clients)

        # Update records for gradients and num of samples
        self.grads = [report.grads for report in reports]
        self.grads = np.array(self.grads)
        self.client_samples = [report.num_samples for report in reports]
        self.client_samples = np.array(self.client_samples)

        # Perform PCA transform if the requested pca dimension > 0
        if self.config.pca_dim > 0:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=self.config.pca_dim)
            self.grads = self.pca.fit_transform(self.grads)

            # Distribute the pca model to all clients
            for client in self.clients:
                client.pca = self.pca

    # Run synchronous federated learning
    def run(self, logger=None):
        import fl_model

        rounds = self.config.server.rounds
        target_accuracy = self.config.fl.target_accuracy
        # reports_path = self.config.paths.reports
        model = self.config.model

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # Perform rounds of federated learning
        T_old = 0.0
        for round in range(1, rounds + 1):
            logging.info('**** Round {}/{} ****'.format(round, rounds))

            # Run the sync federated learning round
            T_new = self.sync_round(T_old, logger)
            logging.info('Round finished at time {} s\n'.format(T_new))

            # Update time
            T_old = T_new

            # Test global model accuracy
            if self.config.clients.do_test:  # Get average accuracy from client reports
                _ = [client.test(self.model) for client in self.clients]
                reports = self.reporting(self.clients)
                test_loss, accuracy = self.accuracy_averaging(reports)
            else:  # Test updated model on server
                testset = self.loader.get_testset()
                batch_size = self.config.fl.batch_size
                testloader = fl_model.get_testloader(testset, batch_size)
                test_loss, accuracy = fl_model.test(self.model, testloader)

            logging.info(
                'Test loss: {} Average accuracy: {:.2f}%'.format(
                    test_loss, 100 * accuracy
                ))

            # Tensorboard logger
            if logger is not None:
                logger.log_value('test_loss', test_loss, int(T_new * 1000))
                logger.log_value('accuracy', accuracy, int(T_new * 1000))

            # Record logger
            self.records.append_record(t=T_new, test_loss=test_loss,
                                       acc=accuracy,
                                       cloud_ca_time=self.ca.asso_time,
                                       total_comm_size=self.total_comm_size)
            for gateway_id in range(len(self.gateways)):
                self.records.append_to_key("gw_cs_time_{}".format(gateway_id),
                                           self.gateway_cs_time[gateway_id])
                self.records.append_to_key("gw_round_time_{}".format(gateway_id),
                                           self.gateway_round_time[gateway_id])
            self.records.save_latest_record(self.config.model_name)

            # Break loop when target accuracy is met
            if model != 'HPWREN' and target_accuracy and \
                    (self.records.get_latest_acc() >= target_accuracy):
                logging.info('Target accuracy reached.')
                break
            elif model == 'HPWREN' and target_accuracy and \
                    (self.records.get_latest_acc() <= target_accuracy):
                logging.info('Target MSE reached.')
                break

            # Adjust client-gateway association
            if round % self.config.server.adjust_round == 0:
                self.conn = self.ca.solve(self.conn_ub, self.grads, self.client_samples,
                                          self.R, self.R_ub, self.config.ca_phi)
                for i in range(len(self.clients)):
                    # Randomly select gateway associations
                    gateway_id_old = self.clients[i].gateway_id
                    gateway_id = np.where(self.conn[i])[0][0]
                    # print(gateway_id)
                    if gateway_id_old != gateway_id:
                        self.gateways[gateway_id_old].remove_client(self.clients[i].client_id)
                        self.gateways[gateway_id].add_client(self.clients[i].client_id)
                        self.clients[i].set_gateway(gateway_id)

        # if reports_path:
        #    with open(reports_path, 'wb') as f:
        #        pickle.dump(self.saved_reports, f)
        #    logging.info('Saved reports: {}'.format(reports_path))

    def sync_round(self, T_old, logger):
        import fl_model  # pylint: disable=import-error

        # Sync the global grads with all gateways
        for gateway in self.gateways:
            gateway.update(self.grads, self.client_samples)

        # Configure gateway to get ready
        _ = [gateway.sync_gateway_configure(self.config) for gateway in self.gateways]

        # Run clients using multithreading for better parallelism
        #threads = [Thread(target=gateway.sync_run(T_old, self.loader, logger))
        #           for gateway in self.gateways]
        #[t.start() for t in threads]
        #[t.join() for t in threads]
        for gateway in self.gateways:
            gateway.semi_async_run(T_old, self.loader, logger)

        # Receive client updates
        reports = self.reporting(self.gateways)

        # Decide the current time
        T_cur = max([report.finish_time for report in reports])

        # Update records of grads and num of samples
        for report in reports:
            self.grads[report.conn_ind] = report.grads[report.conn_ind]
            self.client_samples[report.conn_ind] = report.client_samples[report.conn_ind]
            self.total_comm_size += report.gateway_comm_size + self.config.fl.model_size
            self.gateway_cs_time[report.gateway_id] += report.gateway_cs_time
            self.gateway_round_time[report.gateway_id] += report.gateway_round_time

        # Perform weight aggregation
        logging.info('Cloud aggregating updates')
        updated_weights = self.aggregation(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Extract flattened weights (if applicable)
        #if self.config.paths.reports:
        #    self.save_reports(round, reports)

        # Save updated global model
        saved_model_path = self.config.paths.saved_model
        self.save_model(self.model, saved_model_path)

        return T_cur
