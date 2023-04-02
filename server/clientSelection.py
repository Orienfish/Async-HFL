import random
import math
import numpy as np
import logging
import time

coreset_opt = 'opt' # or 'greedy'

def plot(**kwargs):
    """Plot the histograms of given distributions"""
    import matplotlib.pyplot as plt
    num_plots = len(kwargs)
    for i, (key, value) in enumerate(kwargs.items()):
        plt.subplot(1, num_plots, i+1)
        plt.hist(value)
        plt.title(key)
    plt.show()

class Tier(object):
    """Tier objects for client selection"""
    def __init__(self, client_list, probability, credits):
        self.client_list = client_list
        self.p = probability
        self.credits = credits
        self.mean_loss = 10

class ClientSelection(object):
    """Client selection decision making."""
    def __init__(self, clients, select_type, model_name,
                 thpt_ub, rounds, gamma, alpha, semi_period):
        self.clients = clients
        self.n_clients = len(self.clients)
        self.select_type = select_type
        self.thpt_ub = thpt_ub
        self.rounds = rounds
        self.gamma = gamma
        self.alpha = alpha
        self.semi_period = semi_period
        self.sel_time = 0.0

        # Perform light-weight profiling on clients
        if self.select_type == 'tier':
            self.tiers = self.tier_profiling()

    def update_grads(self, grads, num_samples, client_id='all'):
        # if client_id = 'all', grads: numpy array of grads on all clients, (n, dim of grads)
        #                       num_samples: numpy array of num_samples on all clients, (n,)
        # if client_id = int, grads: numpy array of grads on client #client_id, (n,)
        #                     num_samples: int of num_samples on client #client_id
        if client_id == 'all':  # First-time update the grads mat
            self.grads = grads
            self.num_samples = np.reshape(num_samples, (self.n_clients, 1))
            self.avg_grad = np.sum(
                np.multiply(grads, self.num_samples), axis=0
            ) / np.sum(self.num_samples)  # (n,)

            self.grads_err_mat = np.zeros((self.n_clients, self.n_clients))
            for i in range(self.n_clients):
                self.grads_err_mat[i, :] = np.sum(
                    np.square(self.grads - self.grads[i]), axis=1
                )

        else:  # update grads on client #client_id
            self.avg_grad -= self.num_samples[client_id] / np.sum(num_samples) * \
                             self.grads[client_id, :]
            self.grads[client_id, :] = grads
            self.num_samples[client_id] = num_samples
            self.avg_grad += self.num_samples[client_id] / np.sum(num_samples) * \
                             self.grads[client_id, :]

            self.grads_err_mat[client_id, :] = np.sum(
                np.square(self.grads - self.grads[client_id]), axis=1
            )
            self.grads_err_mat[:, client_id] = self.grads_err_mat[client_id, :]

            # Update disimilarity matrix
            #self.dissimil_mat[client_id, :] = grads @ self.grads.T
            #self.dissimil_mat[:, client_id] = self.dissimil_mat[client_id, :]
            #np.fill_diagonal(self.dissimil_mat, 0.0)

        # print(self.grads_mat)

    def tier_profiling(self):
        #rounds = self.config.fl.rounds
        #clients_per_round = self.config.clients.per_round

        # Run a profiling round to get client delay
        #_ = [client.set_delay() for client in self.clients]

        # Sort clients by delay, fastest first
        sorted_clients = sorted(self.clients, key=lambda c:c.delay)
        for c in sorted_clients:
            print(c.delay)

        # Determine number of tiers
        est_clients_per_round = 5
        m = len(self.clients)/est_clients_per_round

        if m < 1:
            m = 1
        elif m < 5:
            m = math.floor(m)
        elif m <= 10:
            m = 5
        else:
            m = 10

        # Determine the credits / tier
        credits = math.ceil(self.rounds / m)

        # Give equal tier equal probability to begin
        p = 1/m

        # Place clients in each group
        clients_per_group = math.floor(len(self.clients)/m)

        tiers = {}
        for i in range(0, m):
            if i != m-1:
                temp = sorted_clients[clients_per_group * i : clients_per_group * (i+1)]
            else:
                temp = sorted_clients[clients_per_group * i : ]
            tiers[i] = Tier(temp, p, credits)

        return tiers

    def tier_change_prob(self):
        selected_tier = self.last_select_tier
        mean = sum(
            [client.loss for client in self.tiers[selected_tier].client_list])
        mean /= len(self.tiers[selected_tier].client_list)
        self.tiers[selected_tier].mean_loss = mean

        # tiers = [self.tiers[tier] for tier in self.tiers]

        #sort tiers highest loss first
        sorted_tiers = sorted(self.tiers, key=lambda t: self.tiers[t].mean_loss,
                              reverse=True)

        #count tiers with credits left
        credit_cnt = 0
        for tier in sorted_tiers:
            print("Tier Loss" + str(tier) + " : " + str(self.tiers[tier].mean_loss))
            print("Tier Credits" + str(tier) + " : " + str(self.tiers[tier].credits))

            if self.tiers[tier].credits > 0:
                credit_cnt = credit_cnt + 1

        #reset the probability for each tier
        D = credit_cnt * (credit_cnt - 1) / 2

        i = 0
        for tier in sorted_tiers:

            if self.tiers[tier].credits == 0:
                self.tiers[tier].p = 0
                continue
            elif D > 0:
                temp = (credit_cnt-i)/D
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            else:
                temp = credit_cnt -i
                if temp < 0:
                    temp = 0
                self.tiers[tier].p = temp
            print("Tier " + str(tier) + " : " + str(self.tiers[tier].p))
            i = i + 1

    def select(self, cur_thpt):
        # Select k devices to participate in round
        candidates = [c for c in self.clients if c.available]
        # Filter out candidates that consume more thpt than the available thpt
        candidates = [c for c in candidates if c.throughput < (self.thpt_ub - cur_thpt)]
        flag = np.array([c.available and c.throughput < (self.thpt_ub - cur_thpt)
                         for c in self.clients])

        logging.info('select starts!')
        start = time.time()

        if self.select_type == 'divfl':
            sample_clients = []
            while len(candidates) > 0 and cur_thpt < self.thpt_ub:
                # selected: used in divFL, a numpy array of [True or False]
                # indicating whether each client is already selected
                not_selected = np.array([c.available for c in self.clients])
                selected = ~not_selected
                # print(not_selected)

                # cur_G: (not selected, 1), current error in approximating not selected clients
                if np.sum(selected) > 1:  # More than one selected client right now
                    cur_G = np.min(
                        self.grads_err_mat[not_selected][:, selected], axis=1,
                        keepdims=True)
                elif np.sum(selected) == 1:  # Only one selected client right now
                    cur_G = self.grads_err_mat[not_selected, selected]
                else:  # First selection, no client is selected right now
                    cur_G = np.max(self.grads_err_mat, axis=1, keepdims=True)
                # print(cur_G, cur_G.shape)

                # err_rdt: (not selected, not selected), reduction in error if selected one more client
                # err_rdt[i, j]: the reduction in error for approximating i if j is selected
                err_rdt = np.maximum(
                    cur_G - self.grads_err_mat[not_selected][:, not_selected],
                    0.0)
                # print(err_rdt, err_rdt.shape)

                # total_err_rdt: (not selected), total error reduction if select j from non-selected clients
                total_err_rdt = np.sum(err_rdt, axis=0)
                select_client = candidates[np.argmax(total_err_rdt)]
                # print(total_err_rdt, select_client)
                sample_clients.append(select_client)

                # Update client availability status
                select_client.set_unavailable()
                cur_thpt += select_client.throughput
                candidates = [c for c in self.clients if c.available]

            cur_sel_time = time.time() - start
            self.sel_time += cur_sel_time
            logging.info('select time: {}'.format(cur_sel_time))

            return sample_clients  # the function ends here

        else:  # greedy approaches, first sort all candidates according to certain rules

            if self.select_type == 'random':
                # Select clients
                random.shuffle(candidates)

            elif self.select_type == 'high_loss_first':
                # Select the clients with largest loss and random latency
                candidates = sorted(candidates, key=lambda c: c.loss, reverse=True)

            elif self.select_type == 'short_latency_first':
                # Select the clients with short latencies and random loss
                candidates = sorted(candidates, key=lambda c: c.delay)

            elif self.select_type == 'short_latency_high_loss_first':
                # Get the non-negative losses and delays
                losses = np.array([c.loss for c in candidates])
                mean, var = np.mean(losses), np.std(losses)
                losses = (losses - mean) / var
                delays = np.array([c.delay for c in candidates])
                mean, var = np.mean(delays), np.std(delays)
                delays = (delays - mean) / var

                # Sort the clients by jointly consider latency and loss
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: losses[i] - self.gamma * delays[i],
                                    reverse=True)
                print([losses[i] for i in sorted_idx])
                print([self.gamma * delays[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            elif self.select_type == 'tier':
                # Select a tier based on probabilities
                tiers = [num for num in self.tiers]
                tier_prob = [self.tiers[num].p for num in self.tiers]
                selected_tier = random.choices(tiers, weights=tier_prob)[0]
                print('selected_tier: ', selected_tier)
                credits = self.tiers[selected_tier].credits
                while credits == 0:
                    selected_tier = random.choices(tiers, weights=tier_prob)[0]
                    credits = self.tiers[selected_tier].credits

                self.tiers[selected_tier].credits = credits - 1
                self.last_select_tier = selected_tier

                # Select candidates randomly from tier
                candidates = self.tiers[selected_tier].client_list
                random.shuffle(candidates)

            elif self.select_type == 'oort':
                # Get all delays of candidates
                delays = np.array([c.delay for c in candidates])
                flag = (delays > self.semi_period)
                delays_inv = flag * (1 / delays) ** self.alpha + \
                             (~flag * np.ones((len(candidates),)))

                losses = np.square(np.array([c.loss for c in candidates]))
                candidates_flag = np.array([c.available for c in self.clients])
                num_samples = self.num_samples[candidates_flag].reshape((-1))
                losses = num_samples * np.sqrt(np.divide(losses, num_samples))

                # Sort the clients by jointly consider latency and loss
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: losses[i] * delays_inv[i],
                                    reverse=True)
                print([losses[i] for i in sorted_idx])
                print([delays_inv[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            elif 'coreset' in self.select_type:
                 # Get all delays of candidates
                delays = np.array([c.delay for c in candidates])
                delays_inv = (1 / delays) ** self.alpha

                # Compute the gradient similarity and dissimilarity
                # print(np.square(self.grads).sum(axis=1))
                if self.select_type == 'coreset_v1':

                    # Update disimilarity matrix
                    self.dissimil_mat = self.grads @ self.grads.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                    # print(self.dissimil_mat)

                    eta = self.avg_grad @ self.grads[flag].T  # (n,)
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)  # (n,)
                    div = eta + v

                elif self.select_type == 'coreset_v2':

                    # Normalize self.grads
                    grads_normed = self.grads / np.linalg.norm(self.grads, axis=1).reshape((-1, 1))
                    avg_grad_normed = self.avg_grad / np.linalg.norm(self.avg_grad)

                    # Update disimilarity matrix
                    self.dissimil_mat = grads_normed @ grads_normed.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                    # print(self.dissimil_mat)

                    eta = avg_grad_normed @ grads_normed[flag].T  # (n,)
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)  # (n,)

                    loss = np.array([c.loss for c in candidates])
                    div = (eta + v) * loss

                elif self.select_type == 'coreset_v3':

                    # Update disimilarity matrix
                    self.dissimil_mat = self.grads @ self.grads.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                    # print(self.dissimil_mat)

                    eta = self.avg_grad @ self.grads[flag].T  # (n,)
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)  # (n,)

                    loss = np.array([c.loss for c in candidates])
                    div = (eta + v) * loss

                elif self.select_type == 'coreset_v4':

                    # Normalize self.grads
                    grads_normed = self.grads / np.linalg.norm(self.grads, axis=1).reshape((-1, 1))
                    avg_grad_normed = self.avg_grad / np.linalg.norm(
                        self.avg_grad)

                    # Update disimilarity matrix
                    self.dissimil_mat = grads_normed @ grads_normed.T
                    np.fill_diagonal(self.dissimil_mat, 0.0)
                    # print(self.dissimil_mat)

                    eta = avg_grad_normed @ grads_normed[flag].T  # (n,)
                    v = - np.sum(self.dissimil_mat[flag][:, flag], axis=1) / (
                                len(candidates) - 1)  # (n,)

                    div = (eta + v)

                else:
                    raise ValueError(
                        "client select type not implemented: {}".format(
                            self.select_type))

                # Scale div to standard [0, 1] linearly
                max_div, min_div = np.max(div), np.min(div)
                div = (div - min_div) / (max_div - min_div)

                # plot(div=div, loss=losses, delays=delays, delays_inv=delays_inv, c=div*delays_inv)

                # Get the array of throughputs
                thpt = np.array([c.throughput for c in candidates])

                if coreset_opt == 'opt':   # Use gurobi optimal solver
                    import gurobipy as gp
                    from gurobipy import GRB

                    N = len(candidates)
                    thpt = thpt.reshape((1, N))

                    # Create model and variables
                    with gp.Env(empty=True) as env:
                        env.setParam('OutputFlag', 0)
                        env.start()
                        with gp.Model(env=env) as model:
                            model.setParam('MIPGap', 0.1)
                            model.setParam('Timelimit', 100)
                            vars = model.addMVar(shape=N, vtype=GRB.BINARY,
                                                 name='vars')

                            # Set objective
                            c = (div * delays_inv).reshape((1, -1))
                            model.setObjective(c @ vars, GRB.MAXIMIZE)

                            # Each value appears once per row
                            model.addConstr(
                                thpt @ vars <= self.thpt_ub - cur_thpt,
                                name='throughput')

                            # Optimize model
                            model.optimize()

                            client_select = np.array(vars.X, dtype=np.int)
                            sample_clients = [candidates[i] for i in range(N) if
                                              client_select[i] > 0]
                            logging.info('Select: {}'.format(sample_clients))
                            logging.info('avail thpt: {} select thpt: {}'.format(
                                    self.thpt_ub - cur_thpt,
                                    np.sum(thpt * client_select)))
                            logging.info('Select {} out of {}\n'.format(
                                sum(client_select), N))

                            cur_sel_time = time.time() - start
                            self.sel_time += cur_sel_time
                            logging.info('select time: {}'.format(cur_sel_time))

                            return sample_clients  # the function ends here

                # Else, use the greedy heuristic
                # Sort the clients by jointly consider latency and loss
                sorted_idx = sorted(range(len(candidates)),
                                    key=lambda i: div[i] * delays_inv[i] / thpt[i],
                                    reverse=True)
                print([div[i] for i in sorted_idx])
                print([delays_inv[i] for i in sorted_idx])
                candidates = [candidates[i] for i in sorted_idx]

            else:
                raise ValueError(
                    "client select type not implemented: {}".format(self.select_type))

            # Pick the first k clients while satisfying the throughput upperbound
            thpt_list = [c.throughput for c in candidates]
            accum_thpt = [sum(thpt_list[0:i[0]+1]) for i in enumerate(thpt_list)]
            # print([c.delay for c in candidates])
            # print(thpt_list)
            k = len(accum_thpt)
            for index, elem in enumerate(accum_thpt):
                if elem > self.thpt_ub - cur_thpt:
                    k = index
                    break
            sample_clients = candidates[:k]
            logging.info('Select: {}'.format(sample_clients))
            logging.info('Max thpt: {} avail thpt: {} select thpt: {}'.format(accum_thpt[-1] if len(accum_thpt) > 0 else 0,
                                                                       self.thpt_ub - cur_thpt,
                                                                       accum_thpt[k-1] if k > 0 else 0))
            logging.info('Select {} out of {}'.format(k, len(candidates)))

        cur_sel_time = time.time() - start
        self.sel_time += cur_sel_time
        logging.info('select time: {}'.format(cur_sel_time))

        # print(sample_clients)
        return sample_clients