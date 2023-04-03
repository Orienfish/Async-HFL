import logging
import numpy as np
from threading import Thread
from queue import PriorityQueue
import torch
import copy
from .asyncEvent import asyncEvent
from .clientSelection import ClientSelection
import time

class Gateway(object):
    """Gateway on the middle level."""

    def __init__(self, gateway_id, all_clients, config):
        self.gateway_id = gateway_id
        self.all_clients = all_clients  # List of all clients
        self.conn_ind = np.zeros(len(all_clients), dtype=np.bool)
        self.config = config
        self.throughput_ub = config.gateways.throughput_ub
        self.selection = config.selection
        self.cs_gamma = config.cs_gamma_from
        self.cs_alpha = config.cs_alpha
        self.global_rounds = config.server.rounds

        # Keep record of the num_samples and gradients of all global clients
        self.client_samples = None
        self.grads = None

    def add_client(self, client_id):
        self.conn_ind[client_id] = True
        self.clients = [self.all_clients[i] for i in
                        range(len(self.all_clients))
                        if self.conn_ind[i]]

    def remove_client(self, client_id):
        self.conn_ind[client_id] = False
        self.clients = [self.all_clients[i] for i in
                        range(len(self.all_clients))
                        if self.conn_ind[i]]

    # Server interactions
    def download(self, argv):
        # Download from the server.
        try:
            return copy.deepcopy(argv)
        except:
            return argv

    def upload(self, argv):
        # Upload to the server
        try:
            return copy.deepcopy(argv)
        except:
            return argv

    def get_report(self):
        # Report results to server.
        return self.upload(self.report)

    def update(self, grads, client_samples):
        self.grads = self.download(grads)  # numpy array
        self.client_samples = self.download(client_samples)  # numpy array

    def set_delay_to_cloud(self, delay):
        # Set the link speed and delay for the upcoming run
        self.delay = delay

    def sync_gateway_configure(self, config):
        import fl_model  # pylint: disable=import-error

        # Download most recent global model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global'
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.sync_save_gateway_model(self.model, saved_model_path)

        # If the server is in the semi-async mode, create a semi-async queue
        # each time the gateway is configured for a new run
        self.semi_qEvents = PriorityQueue()

    def async_gateway_configure(self, config, download_time, cur_round):
        import fl_model  # pylint: disable=import-error

        # Download most recent global model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_' + '{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.async_save_gateway_model(self.model, saved_model_path,
                                      download_time)

        self.global_download_time = download_time  # Used for regularize global model

        # Adjust the gamma rate during client selection
        self.adjust_client_selection_gamma(cur_round)

    def adjust_client_selection_gamma(self, cur_round):
        # Adjust the gamma weight to balance learning utility and delays
        # during coreset client selection
        # Only used in async-hier setup
        self.cs_gamma = self.config.cs_gamma_from - \
                        (self.config.cs_gamma_from - self.config.cs_gamma_to) * \
                        cur_round / self.global_rounds

    def rflha_gateway_configure(self, config, download_time):
        import fl_model  # pylint: disable=import-error

        # Download most recent global model
        saved_model_path = config.paths.saved_model
        path = saved_model_path + '/global_' + '{}'.format(download_time)
        self.model = fl_model.Net()
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info('Gateway {} load global model: {}'.format(
            self.gateway_id, path))

        self.sync_save_gateway_model(self.model, saved_model_path)

    def sync_client_configure(self, sample_clients):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuration on client
            client.sync_client_configure(config)

    def async_client_configure(self, sample_clients, gateway_download_time):
        loader_type = self.config.loader
        loading = self.config.data.loading

        if loading == 'dynamic':
            # Create shards if applicable
            if loader_type == 'shard':
                self.loader.create_shards()

        # Configure selected clients for federated learning task
        for client in sample_clients:
            if loading == 'dynamic':
                self.set_client_data(client)  # Send data partition to client

            # Extract config for client
            config = self.config

            # Continue configuration on client
            client.async_client_configure(config, gateway_download_time)

    # Run synchronous federated learning
    def sync_run(self, T_old, loader, logger):
        import fl_model  # pylint: disable=import-error

        rounds = self.config.gateways.rounds

        # Create client selection class and update grads for all associated clients
        self.cs = ClientSelection(self.clients, self.selection, self.config.model_name,
                                  self.throughput_ub, rounds, self.cs_gamma,
                                  self.cs_alpha, self.config.semi_period)
        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])

        # Perform rounds of federated learning
        gateway_comm_size = 0
        T_start = T_old
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))

            # Run the sync federated learning round
            T_new, comm_size = self.sync_round(T_start)
            gateway_comm_size += comm_size
            logging.info('Gw {} Round finished at time {} s\n'.format(self.gateway_id,
                                                                      T_new))

            # Test
            testset = loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            test_loss, accuracy = fl_model.test(self.model, testloader)

            logging.info('test loss: {} acc: {}'.format(test_loss, accuracy))
            logger.log_value('gw{}_accuracy'.format(self.gateway_id),
                             accuracy, int(T_new * 1000))

            # Update time
            T_start = T_new

        # Generate report for server
        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_new + self.delay,
                             T_new - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, test_loss,
                             accuracy)
        return T_new + self.delay

    def sync_round(self, T_old):
        import fl_model  # pylint: disable=import-error

        # Select clients to participate in the round
        self.throughput = 0

        sample_clients = self.cs.select(cur_thpt=self.throughput)

        self.sync_client_configure(sample_clients)

        _ = [client.set_unavailable() for client in sample_clients]
        self.throughput = sum([client.throughput for client in sample_clients])

        logging.info('Gw {} throughput {} kB/s'.format(self.gateway_id,
                                                       self.throughput))

        # Use the max delay in all sample clients as the delay in sync round
        max_delay = 0
        if len(sample_clients) > 0:
            max_delay = max([c.delay for c in sample_clients])

        # Run clients using multithreading for better parallelism
        if self.config.model not in ['CIFAR-10', 'FEMNIST']:
            threads = [Thread(target=client.run) for client in
                       sample_clients]
            [t.start() for t in threads]
            [t.join() for t in threads]
        else:  # Use sequential execution because ResNet is too big
            _ = [client.run() for client in sample_clients]

        T_cur = T_old + max_delay  # Update current time

        # Receive client updates
        reports = self.reporting(sample_clients)

        # Update grads and num of samples
        for report in reports:
            self.grads[report.client_id] = report.grads
            self.client_samples[report.client_id] = report.num_samples

        # Perform weight aggregation
        logging.debug('Gw {} aggregating updates'.format(self.gateway_id))
        updated_weights = fl_model.extract_weights(self.model)
        if len(sample_clients) > 0:
            updated_weights = self.federated_averaging(reports)

        # Load updated weights
        fl_model.load_weights(self.model, updated_weights)

        # Save updated global model
        saved_model_path = self.config.paths.saved_model
        self.sync_save_gateway_model(self.model, saved_model_path)

        # Re-enable the selected client
        _ = [client.set_available() for client in sample_clients]

        # Update client selection info
        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])
        elif self.selection == 'tier' and len(sample_clients) > 0:
            self.cs.tier_change_prob()

        return T_cur, self.config.fl.model_size * len(sample_clients)

    def async_run(self, T_old, loader, logger):
        """Run one async round until the slowest client finish one round"""
        import fl_model

        rounds = self.config.gateways.rounds

        # Init async parameters
        self.alpha = self.config.async_params.alpha
        self.gl_alpha = self.config.async_params.gl_alpha
        self.rou = self.config.async_params.rou
        self.staleness_func = self.config.async_params.staleness_func

        # Create client selection class and update grads for all associated clients
        self.cs = ClientSelection(self.clients, self.selection, self.config.model_name,
                                  self.throughput_ub, rounds, self.cs_gamma,
                                  self.cs_alpha, self.config.semi_period)
        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])


        # Initial sample
        # Select clients to participate in the round
        self.throughput = 0
        sample_clients = self.cs.select(self.throughput)

        # Create a queue at the server to store gateway update events
        qEvents = PriorityQueue()

        # async client configuration
        logging.debug('Gw {} T_old: {}'.format(self.gateway_id, T_old))
        self.async_client_configure(sample_clients, T_old)

        for client in sample_clients:
            # Disable selected clients to prevent from re-selection
            # client.set_delay_to_gateway()
            client.set_unavailable()

            # Create event and put into the priority queue
            new_event = asyncEvent(client, 0, T_old, client.delay)
            qEvents.put(new_event)

        self.throughput = sum([client.throughput for client in sample_clients])

        # Perform rounds of async federated learning
        gateway_comm_size = 0
        self.cs.sel_time = 0
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))

            event = qEvents.get()
            select_client = event.client

            # Run on the client
            select_client.run(reg=True, rou=self.rou)
            T_cur = event.download_time + select_client.delay
            gateway_comm_size += self.config.fl.model_size

            # Request report on weights, loss, delay, throughput
            report = select_client.get_report()

            # Update grads
            self.grads[report.client_id] = report.grads
            self.client_samples[report.client_id] = report.num_samples

            # Re-enable the selected client
            select_client.set_available()
            self.throughput -= select_client.throughput

            # Perform weight aggregation
            logging.info('Aggregating updates on gateway {} from clients {} at {}'.format(
                self.gateway_id, select_client.client_id, T_cur))
            staleness = round - event.download_round
            updated_weights = self.federated_async(report, staleness)

            # Load updated weights and save as the latest model
            fl_model.load_weights(self.model, updated_weights)
            saved_model_path = self.config.paths.saved_model
            self.async_save_gateway_model(self.model, saved_model_path, T_cur)

            # Test
            testset = loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            test_loss, accuracy = fl_model.test(self.model, testloader)

            logging.info('test loss: {} acc: {}\n'.format(test_loss, accuracy))
            logger.log_value('gw{}_accuracy'.format(self.gateway_id),
                             accuracy, int(T_cur * 1000))

            # Select one more client and insert to qEvents
            if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
                self.cs.update_grads(self.grads[self.conn_ind],
                                     self.client_samples[self.conn_ind])

            new_clients = self.cs.select(self.throughput)

            # async client configuration
            self.async_client_configure(new_clients, T_cur)

            for client in new_clients:
                # Set delay and unavailability
                # client.set_delay_to_gateway()
                client.set_unavailable()

                # Create event and put into the priority queue
                new_event = asyncEvent(client, round, T_cur, client.delay)
                qEvents.put(new_event)

                self.throughput += client.throughput

        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_cur + self.delay,
                             T_cur - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, test_loss,
                             accuracy)

        # Re-enable all clients
        _ = [client.set_available() for client in self.clients]

        return T_cur + self.delay

    def semi_async_run(self, T_old, loader, logger):
        """Run one async round until the slowest client finish one round"""
        import fl_model

        # Init semi async parameters
        rounds = self.config.gateways.rounds
        semi_period = self.config.semi_period
        self.llambda = self.config.async_params.llambda

        # Create client selection class and update grads for all associated clients
        self.cs = ClientSelection(self.clients, self.selection, self.config.model_name,
                                  self.throughput_ub, rounds, self.cs_gamma,
                                  self.cs_alpha, semi_period)
        if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
            self.cs.update_grads(self.grads[self.conn_ind],
                                 self.client_samples[self.conn_ind])

        # Initial sample
        # Select clients to participate in the round
        self.throughput = 0
        sample_clients = self.cs.select(self.throughput)
        print('Select client delays: {}'.format([client.delay for client in sample_clients]))

        for client in sample_clients:
            # Disable selected clients to prevent from re-selection
            # client.set_delay_to_gateway()
            client.set_unavailable()

            # Run client
            self.sync_client_configure([client])
            client.run()

            # Create event and put into the priority queue
            new_event = asyncEvent(client, 0, T_old, client.delay)
            self.semi_qEvents.put(new_event)

        self.throughput = sum([client.throughput for client in sample_clients])

        # Perform rounds of semiasync federated learning
        gateway_comm_size = 0
        self.cs.sel_time = 0
        for round in range(1, rounds + 1):
            logging.info('**** Gw {} Round {}/{} ****'.format(self.gateway_id,
                                                              round, rounds))

            normal_reports, straggler_reports = [], []
            max_staleness = 0

            while not self.semi_qEvents.empty():  # Pops all events within the next period
                event = self.semi_qEvents.get()
                select_client = event.client
                T_cur = event.download_time + select_client.delay

                # Check if the current event is received within the current period
                if T_cur > T_old + round * semi_period:
                    # If the current event cannot be processed within the current period
                    # put the event back to the queue, and break loop
                    self.semi_qEvents.put(event)
                    break

                # If the code reaches here, if means the event is received successfully
                gateway_comm_size += self.config.fl.model_size

                # Request report on weights, loss, delay, throughput
                # Append the report to normal or straggler reports
                report = select_client.get_report()
                max_staleness = max(max_staleness, round - event.download_round)
                if event.download_round == round - 1:  # Normal clients
                    normal_reports.append(report)
                else:
                    straggler_reports.append(report)

                # Update grads
                self.grads[report.client_id] = report.grads
                self.client_samples[report.client_id] = report.num_samples

                # Re-enable the selected client
                select_client.set_available()
                self.throughput -= select_client.throughput

            # Perform weight aggregation
            logging.info('Aggregating updates on gateway {} from normal clients {} '
                         'and straggler clients {}'.format(self.gateway_id,
                                                           [report.client_id for report in normal_reports],
                                                           [report.client_id for report in straggler_reports]))
            updated_weights = self.federated_semi_async(normal_reports, straggler_reports,
                                                        max_staleness)

            # Load updated weights and save as the latest model
            fl_model.load_weights(self.model, updated_weights)
            saved_model_path = self.config.paths.saved_model
            self.sync_save_gateway_model(self.model, saved_model_path)

            # Test
            testset = loader.get_testset()
            batch_size = self.config.fl.batch_size
            testloader = fl_model.get_testloader(testset, batch_size)
            test_loss, accuracy = fl_model.test(self.model, testloader)

            logging.info('test loss: {} acc: {}\n'.format(test_loss, accuracy))
            logger.log_value('gw{}_accuracy'.format(self.gateway_id),
                             accuracy, int(T_cur * 1000))

            # Select one more client and insert to qEvents
            if self.selection in ['divfl', 'oort'] or 'coreset' in self.selection:
                self.cs.update_grads(self.grads[self.conn_ind],
                                     self.client_samples[self.conn_ind])

            new_clients = self.cs.select(self.throughput)

            for client in new_clients:
                # Set delay and unavailability
                # client.set_delay_to_gateway()
                client.set_unavailable()

                # Create event and put into the priority queue
                new_event = asyncEvent(client, round, T_cur, client.delay)
                self.semi_qEvents.put(new_event)

                self.throughput += client.throughput

        gateway_weights = fl_model.extract_weights(self.model)
        total_samples = np.sum(self.client_samples[self.conn_ind])
        T_cur = T_old + rounds * semi_period

        self.report = Report(self, gateway_weights, self.grads,
                             self.client_samples,
                             total_samples, T_cur + self.delay,
                             T_cur - T_old + self.delay,
                             self.cs.sel_time, gateway_comm_size, test_loss,
                             accuracy)

        # Re-enable all clients
        _ = [client.set_available() for client in self.clients]

        return T_cur + self.delay

    def federated_averaging(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        updates = self.extract_client_updates(reports)

        # Extract total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        avg_update = [torch.zeros(x.size())  # pylint: disable=no-member
                      for _, x in updates[0]]
        for i, update in enumerate(updates):
            num_samples = reports[i].num_samples
            for j, (_, delta) in enumerate(update):
                # Use weighted average by number of samples
                avg_update[j] += delta * (num_samples / total_samples)

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Load updated weights into model
        updated_weights = []
        for i, (name, weight) in enumerate(baseline_weights):
            updated_weights.append((name, weight + avg_update[i]))

        return updated_weights

    def federated_async(self, report, staleness):
        import fl_model  # pylint: disable=import-error

        # Extract updates from reports
        weights = self.extract_client_weights([report])[0]

        # Extract baseline model weights - latest model
        baseline_weights = fl_model.extract_weights(self.model)

        # Calculate the staleness-aware weights
        alpha_t = self.gl_alpha * self.staleness(staleness)
        if self.config.server.mode == 'pureasync':
            alpha_t = 1.0  # Naively extending two laters to three layers
        logging.info('{} alpha: {} staleness: {} alpha_t: {}'.format(
            self.staleness_func, self.gl_alpha, staleness, alpha_t
        ))

        # Load updated weights into model
        updated_weights = []
        for i, (t1, t2) in enumerate(zip(baseline_weights, weights)):
            assert t1[0] == t2[0], "weights names do not match!"
            name, old_weight, new_weight = t1[0], t1[1], t2[1]
            updated_weights.append(
                (name, (1 - alpha_t) * old_weight + alpha_t * new_weight)
            )

        return updated_weights

    def federated_semi_async(self, normal_reports, straggler_reports, max_staleness):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights - latest model
        baseline_weights = fl_model.extract_weights(self.model)

        # 1. Perform FecAvg-style aggregation on reports from normal devices
        # Extract updates from normal devices' reports
        weights = self.extract_client_weights(normal_reports)
        total_samples = sum([report.num_samples for report in normal_reports])

        # Perform weighted averaging
        if len(normal_reports) > 0:
            normal_avg_weights = [torch.zeros(x.size())  # pylint: disable=no-member
                                  for _, x in weights[0]]
            for i, weight in enumerate(weights):
                num_samples = normal_reports[i].num_samples
                for j, (_, w) in enumerate(weight):
                    # Use weighted average by number of samples
                    normal_avg_weights[j] += w * (num_samples / total_samples)

        # 2. Perform approximation on reports from stragglers
        # Extract weights from stragglers' reports
        weights = self.extract_client_weights(straggler_reports)
        total_samples = sum([report.num_samples for report in straggler_reports])

        if len(straggler_reports) > 0:
            # Calculate the staleness-aware weights
            straggler_avg_weights = [torch.zeros(x.size())  # pylint: disable=no-member
                                     for _, x in weights[0]]
            for i, weight in enumerate(weights):
                num_samples = straggler_reports[i].num_samples
                for j, (n, w) in enumerate(weight):
                    # Use weighted average by number of samples
                    straggler_avg_weights[j] += w * (num_samples / total_samples)

        # 3. Combine the two updated weights Load updated weights into model
        # Decide the staleness-aware parameters lambda_t
        lambda_t = self.llambda * np.exp(- max_staleness)

        updated_weights = []
        if len(normal_reports) > 0 and len(straggler_reports) > 0:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append(
                    (name, (1 - lambda_t) * normal_avg_weights[i] +
                     lambda_t * straggler_avg_weights[i])
                )
        elif len(normal_reports) > 0:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append((name, normal_avg_weights[i]))
        elif len(straggler_reports) > 0:
            for i, (name, weight) in enumerate(baseline_weights):
                updated_weights.append((name, straggler_avg_weights[i]))
        else:
            updated_weights = baseline_weights  # use baseline weights

        return updated_weights

    def accuracy_averaging(self, reports):
        # Get total number of samples
        total_samples = sum([report.num_samples for report in reports])

        # Perform weighted averaging
        accuracy = 0
        for report in reports:
            accuracy += report.accuracy * (report.num_samples / total_samples)

        return accuracy

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

    # Report aggregation
    def extract_client_updates(self, reports):
        import fl_model  # pylint: disable=import-error

        # Extract baseline model weights
        baseline_weights = fl_model.extract_weights(self.model)

        # Extract weights from reports
        weights = [report.weights for report in reports]

        # Calculate updates from weights
        updates = []
        for weight in weights:
            update = []
            for i, (name, w) in enumerate(weight):
                bl_name, baseline = baseline_weights[i]

                # Ensure correct weight is being updated
                assert name == bl_name

                # Calculate update
                delta = w - baseline
                update.append((name, delta))
            updates.append(update)

        return updates

    def extract_client_weights(self, reports):
        # Extract weights from reports
        weights = [report.weights for report in reports]

        return weights

    def extract_client_grads(self, reports):
        # Extract weights from reports
        grads = [report.grads for report in reports]

        return grads

    # Server operations
    @staticmethod
    def flatten_weights(weights):
        # Flatten weights into vectors
        weight_vecs = []
        for _, weight in weights:
            weight_vecs.extend(weight.flatten().tolist())

        return np.array(weight_vecs)

    def reporting(self, sample_clients):
        # Recieve reports from sample clients
        reports = [client.get_report() for client in sample_clients]

        logging.info('Reports received: {}'.format(len(reports)))
        assert len(reports) == len(sample_clients)

        return reports

    def sync_save_gateway_model(self, model, path):
        path += '/gateway{}'.format(self.gateway_id)
        torch.save(model.state_dict(), path)
        logging.debug('Saved global model: {}'.format(path))

    def async_save_gateway_model(self, model, path, download_time):
        path += '/gateway{}_{}'.format(self.gateway_id, download_time)
        torch.save(model.state_dict(), path)
        logging.debug('Saved gateway {} model: {}'.format(self.gateway_id, path))


class Report(object):
    """Federated learning client report."""

    def __init__(self, gateway, weights, grads, client_samples,
                 total_samples, finish_time, gateway_round_time,
                 gateway_cs_time, total_comm_size,
                 test_loss, accuracy):
        self.gateway_id = gateway.gateway_id
        self.conn_ind = gateway.conn_ind
        self.weights = weights
        self.grads = grads  # numpy matrix
        self.client_samples = client_samples  # numpy array
        self.num_samples = total_samples
        self.finish_time = finish_time
        self.gateway_round_time = gateway_round_time
        self.gateway_cs_time = gateway_cs_time
        self.gateway_comm_size = total_comm_size
        self.test_loss = test_loss
        self.accuracy = accuracy

