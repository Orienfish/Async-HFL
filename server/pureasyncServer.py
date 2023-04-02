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

class PureAsyncServer(SyncServer):
    """Asynchronous federated learning server."""

    def load_model(self):
        import fl_model  # pylint: disable=import-error

        saved_model_path = self.config.paths.saved_model
        model_type = self.config.model

        logging.info('Model: {}'.format(model_type))

        # Set up global model
        self.model = fl_model.Net()
        self.async_save_model(self.model, saved_model_path, 0.0)

        # Extract flattened weights (if applicable)
        if self.config.paths.reports:
            self.saved_reports = {}
            self.save_reports(0, [])  # Save initial model

    # Warmup the clients queue by calling all clients
    def server_warmup(self):
        logging.info('Server warmup ...')

        # Init the server and state for the current episode
        _ = [client.async_global_configure(self.config, 0.0) for client in
             self.clients]

        if self.config.model not in ['CIFAR-10', 'FEMNIST', 'Shakespeare']:
            threads = [Thread(target=client.run(reg=True, rou=self.config.async_params.rou))
                       for client in self.clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:  # Use sequential execution because ResNet is too big
            _ = [client.run(reg=True, rou=self.config.async_params.rou) for client in self.clients]

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

    # Run asynchronous federated learning
    def run(self, logger=None):
        import fl_model  # pylint: disable=import-error
        rounds = self.config.server.rounds
        target_accuracy = self.config.fl.target_accuracy
        # reports_path = self.config.paths.reports
        model = self.config.model

        # Init async parameters
        self.gl_alpha = self.config.async_params.gl_alpha
        self.staleness_func = self.config.async_params.staleness_func

        if target_accuracy:
            logging.info('Training: {} rounds or {}% accuracy\n'.format(
                rounds, 100 * target_accuracy))
        else:
            logging.info('Training: {} rounds\n'.format(rounds))

        # qEvents is a queue for all async events
        # Each event is one gateway completing one local update, which is one round
        # Different from the clients, since we don't know how long it takes
        # to complete a gateway round, we need to run it first
        qEvents = PriorityQueue()
        next_gw_agg = []
        for gateway in self.gateways:
            # Sync the global grads with all gateways
            gateway.update(self.grads, self.client_samples)

            # Configure the gateway: load async global model and save as
            # async gateway model
            gateway.async_gateway_configure(self.config, 0.0, 0)

            # Run gateway async and get the new time
            T_new = gateway.async_run(0.0, self.loader, logger)

            next_gw_agg.append(T_new)

            # Create event and put into the priority queue
            new_event = asyncEvent(gateway, 0, 0.0, T_new)
            qEvents.put(new_event)

        # Perform rounds of async federated learning
        for round in range(1, rounds + 1):
            logging.info('\n**** Round {}/{} ****'.format(round, rounds))

            event = qEvents.get()
            select_gateway = event.client
            T_cur = event.aggregate_time  # Update current time
            print(next_gw_agg)

            # Receive client updates
            report = select_gateway.get_report()
            logging.info(
                'Select gateway {}, time {} s, test loss: {}, accuracy: {}'.format(
                    select_gateway.gateway_id, T_cur, report.test_loss,
                    report.accuracy))

            # Update records of grads and num of samples
            self.grads[report.conn_ind] = report.grads[report.conn_ind]
            self.client_samples[report.conn_ind] = report.client_samples[
                report.conn_ind]
            self.total_comm_size += report.gateway_comm_size + self.config.fl.model_size
            self.gateway_cs_time[
                report.gateway_id] += report.gateway_cs_time
            self.gateway_round_time[
                report.gateway_id] += report.gateway_round_time

            # Perform weight aggregation
            # logging.info('Cloud aggregating updates')
            staleness = round - event.download_round
            updated_weights = self.aggregation(report, staleness)

            # Load updated weights
            fl_model.load_weights(self.model, updated_weights)

            # Extract flattened weights (if applicable)
            # if self.config.paths.reports:
            #    self.save_reports(round, [report])

            # Save updated global model
            saved_model_path = self.config.paths.saved_model
            self.async_save_model(self.model, saved_model_path, T_cur)

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
                'T_cur: {} Test loss: {} Average accuracy: {:.2f}%\n'.format(
                    T_cur, test_loss, 100 * accuracy
                ))

            # Tensorboard logger
            if logger is not None:
                logger.log_value('test_loss', test_loss, int(T_cur * 1000))
                logger.log_value('accuracy', accuracy, int(T_cur * 1000))
                logger.log_value('cs_gamma', self.gateways[0].cs_gamma,
                                 int(T_cur * 1000))

            # Record logger
            self.records.append_record(t=T_cur, test_loss=test_loss,
                                       acc=accuracy,
                                       cloud_ca_time=self.ca.asso_time,
                                       total_comm_size=self.total_comm_size)
            for gateway_id in range(len(self.gateways)):
                self.records.append_to_key(
                    "gw_cs_time_{}".format(gateway_id),
                    self.gateway_cs_time[gateway_id])
                self.records.append_to_key(
                    "gw_round_time_{}".format(gateway_id),
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
                self.conn = self.ca.solve(self.conn_ub, self.grads,
                                          self.client_samples,
                                          self.R, self.R_ub,
                                          self.config.ca_phi)
                for i in range(len(self.clients)):
                    # Select gateway associations
                    gateway_id_old = self.clients[i].gateway_id
                    gateway_id = np.where(self.conn[i])[0][0]
                    # print(gateway_id)
                    if gateway_id_old != gateway_id:
                        self.gateways[gateway_id_old].remove_client(
                            self.clients[i].client_id)
                        self.gateways[gateway_id].add_client(
                            self.clients[i].client_id)
                        self.clients[i].set_gateway(gateway_id)

            # Configure the gateway: load async global model and save as
            # async gateway model
            select_gateway.async_gateway_configure(self.config, T_cur,
                                                   round)

            # Run gateway async and get the new time
            T_new = select_gateway.async_run(T_cur, self.loader, logger)

            next_gw_agg[select_gateway.gateway_id] = T_new

            # Create event and put into the priority queue
            new_event = asyncEvent(select_gateway, round, T_cur,
                                   T_new - T_cur)
            qEvents.put(new_event)

        # if reports_path:
        #    with open(reports_path, 'wb') as f:
        #        pickle.dump(self.saved_reports, f)
        #    logging.info('Saved reports: {}'.format(reports_path))

        # Remove outdated and useless model
        saved_model_path = self.config.paths.saved_model
        self.rm_old_models(saved_model_path, T_cur + 1.0)

    def aggregation(self, reports, staleness=None):
        return self.federated_async(reports, staleness)

    def extract_client_weights(self, reports):
        # Extract weights from reports
        weights = [report.weights for report in reports]

        return weights

    def federated_async(self, report, staleness):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        weights = self.extract_client_weights([report])[0]

        # Extract baseline model weights - latest model
        baseline_weights = fl_model.extract_weights(self.model)

        # Calculate the staleness-aware weights
        alpha_t = self.gl_alpha * self.staleness(staleness)
        logging.info('{} alpha: {} staleness: {} alpha_t: {}'.format(
            self.staleness_func, self.gl_alpha, staleness, alpha_t
        ))

        # Load updated weights into model
        updated_weights = []
        for i, (t1, t2) in enumerate(zip(baseline_weights, weights)):
            assert t1[0] == t2[0], "weights names do not match!"
            name, old_weight = t1[0], t1[1]
            new_weight = t2[1]
            updated_weights.append(
                (name, (1 - alpha_t) * old_weight + alpha_t * new_weight)
            )

        return updated_weights

    def staleness(self, staleness):
        if self.staleness_func == "constant":
            return 1
        elif self.staleness_func == "polynomial":
            a = 0.5
            return pow(staleness+1, -a)
        elif self.staleness_func == "hinge":
            a, b = 10, 4
            if staleness <= b:
                return 1
            else:
                return 1 / (a * (staleness - b) + 1)

    def async_save_model(self, model, path, download_time):
        path += '/global_' + '{}'.format(download_time)
        torch.save(model.state_dict(), path)
        logging.info('Saved global model: {}'.format(path))

    def rm_old_models(self, path, cur_time):
        for filename in os.listdir(path):
            try:
                model_time = float(filename.split('_')[1])
                if model_time < cur_time:
                    os.remove(os.path.join(path, filename))
                    logging.info('Remove model {}'.format(filename))
            except Exception as e:
                logging.debug(e)
                continue