import sys
import numpy as np
import logging
import pickle
from threading import Thread
import time
from .server import Server
from .gateway import Gateway
from .record import Record
from .clientAssociation import ClientAssociation

class SyncServer(Server):
    """Synchronous federated learning server."""

    def boot(self):
        logging.info('Booting {} server...'.format(self.config.server.mode))

        model_path = self.config.paths.model
        total_clients = self.config.clients.total
        total_gateways = self.config.gateways.total

        # Add fl_model to import path
        sys.path.append(model_path)

        # Basic setup
        self.load_data()
        self.load_model()
        if self.config.loader != 'leaf':
            num_clients = self.config.clients.total
            self.make_clients(num_clients)
        else:
            num_clients = min(self.config.clients.total, self.loader.num_clients)
            self.make_clients_leaf(num_clients)
        self.make_gateways(total_gateways)

        # Warm up server to collect from all clients
        self.server_warmup()

        # Setup the links
        self.set_link()

        for gateway in self.gateways:
            logging.info('Gw {}: {} clients from '
                         'total feasible clients of {}'.format(
                gateway.gateway_id,
                self.conn.sum(axis=0)[gateway.gateway_id],
                self.conn_ub.sum(axis=0)[gateway.gateway_id]
            ))

        # Init self accuracy records
        self.records = Record("t", "test_loss", "acc", "total_comm_size",
                              "cloud_ca_time")
        for gateway_id in range(total_gateways):
            self.records.insert_key("gw_cs_time_{}".format(gateway_id),
                                    "gw_round_time_{}".format(gateway_id))
        self.total_comm_size = 0.0
        self.gateway_cs_time = [0.0 for _ in range(total_gateways)]
        self.gateway_round_time = [0.0 for _ in range(total_gateways)]

    def make_gateways(self, num_gws):
        gateways = []
        for gateway_id in range(num_gws):
            # Create new gateway
            new_gw = Gateway(gateway_id, self.clients, self.config)
            gateways.append(new_gw)

        self.gateways = gateways

    def set_link(self):
        model_size = self.config.fl.model_size

        if self.config.delay_mode == 'nycmesh':
            # Set sensor-gateway connection and its delay
            delay_sr_to_gw = np.loadtxt(self.config.delays.gateway_client, delimiter=',')[:, :self.config.gateways.total]
            delay_sr = np.loadtxt(self.config.delays.comp_time, delimiter=',')

            # Filter out the rows that are all zero - not connect to any gateway
            valid_sr_id = ~np.all(delay_sr_to_gw == 0, axis=1)
            delay_sr_to_gw = delay_sr_to_gw[valid_sr_id][:self.config.clients.total]
            delay_sr = delay_sr[valid_sr_id][:self.config.clients.total]
            self.conn_ub = (delay_sr_to_gw > 0).astype(np.int)

            assert len(self.clients) <= delay_sr_to_gw.shape[0], \
                "More clients than the rows in the provided delay matrix!"
            assert len(self.gateways) <= delay_sr_to_gw.shape[1], \
                "More gateways than the columns in the provided delay matrix!"

        elif self.config.delay_mode == 'uniform':
            # Sparse ratio indicate the portion of connectable links
            # Hence if the random generated number is less than sparse_ratio, we say the link is
            # connectable, i.e., conn_ub[i,j] = 1
            self.conn_ub = (np.random.uniform(size=(len(self.clients), len(self.gateways))) <
                            self.config.link.sparse_ratio).astype(np.int)

            # Make sure all rows have at least one nonzero item
            zero_rows_flag = (self.conn_ub.sum(axis=1) < .1)
            print(zero_rows_flag)
            if np.sum(zero_rows_flag) > 0:  # If zero row exists
                self.conn_ub[zero_rows_flag, 0] = 1

            speed_sr_to_gw = np.random.uniform(low=self.config.link.min,
                                               high=self.config.link.max,
                                               size=(len(self.clients), len(self.gateways)))
            delay_sr_to_gw = self.conn_ub * (model_size / speed_sr_to_gw)

            delay_sr = np.random.uniform(low=self.config.comp_time.min,
                                         high=self.config.comp_time.max,
                                         size=len(self.clients))
        else:
            raise ValueError(
                    "delay mode not implemented: {}".format(self.config.delay_mode))

        # Create client-gateway association solver
        if self.config.loader == 'bias':
            pref = [client.pref for client in self.clients]
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name,
                                        pref=pref, labels=self.labels)
        elif self.config.loader == 'noniid':
            cls_num = [client.cls_num for client in self.clients]
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name,
                                        cls_num=cls_num, labels=self.labels)
        else:
            self.ca = ClientAssociation(self.config.association,
                                        self.config.model_name)

        # Get the throughput per link and the throughput upperbound per gateway
        self.est_total_delay = delay_sr_to_gw + delay_sr.reshape((-1, 1)) + 1e-10
        self.est_total_delay = self.est_total_delay
        self.R = np.divide(model_size, self.est_total_delay)
        self.R_ub = np.array([self.config.gateways.throughput_ub] * self.config.gateways.total)
        self.conn = self.ca.solve(self.conn_ub, self.grads, self.client_samples,
                                  self.R, self.R_ub, self.config.ca_phi)

        for i in range(len(self.clients)):
            # Randomly select gateway associations
            gateway_id = np.where(self.conn[i])[0][0]
            # print(gateway_id)
            self.gateways[gateway_id].add_client(self.clients[i].client_id)
            self.clients[i].set_gateway(gateway_id)

            comm_delay = delay_sr_to_gw[i, gateway_id]
            comp_delay = delay_sr[i]

            if self.config.delay_mode == 'nycmesh':
                self.clients[i].set_delay_to_gateway(comm_delay, comp_delay,
                                                     model_size)
            elif self.config.delay_mode == 'uniform':
                speed_mean = speed_sr_to_gw[i, gateway_id]
                self.clients[i].set_delay_to_gateway(comm_delay, comp_delay,
                                                     model_size, speed_mean,
                                                     self.config.link.std,
                                                     self.config.comp_time.std)
            else:
                raise ValueError(
                    "delay mode not implemented: {}".format(self.config.delay_mode))

        print('Client to gateway delay distribution: {}'.format(
            [client.delay for client in self.clients]))

        # Set gateway-cloud delay
        delay_gw_to_cl = np.loadtxt(self.config.delays.cloud_gateway, delimiter=',')
        for i in range(len(self.gateways)):
            self.gateways[i].set_delay_to_cloud(delay_gw_to_cl[i])

        print('Gateway to cloud delay distribution: {}'.format(
            [gateway.delay for gateway in self.gateways]))

    # Warmup the clients queue by calling all clients
    def server_warmup(self):
        logging.info('Server warmup ...')
        # Init the server and state for the current episode
        _ = [client.sync_global_configure(self.config) for client in self.clients]
        if self.config.model not in ['CIFAR-10', 'FEMNIST']:
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

        #if reports_path:
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
            gateway.sync_run(T_old, self.loader, logger)

        # Receive client updates
        reports = self.reporting(self.gateways)

        # Decide the current time
        T_cur = max([report.finish_time for report in reports])

        # Update records of grads, num of samples, total comm model size
        # and execution time
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

