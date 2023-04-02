import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class Record(object):
    """Accuracy records."""
    def __init__(self, *args):
        self.record = {}
        for arg in args:
            self.record[arg] = []
        self.alpha = 0.9  # decay factor for smoothing the acc curve

    def insert_key(self, *args):
        for arg in args:
            self.record[arg] = []

    def append_record(self, **kwargs):
        for arg in kwargs:
            self.record[arg].append(kwargs[arg])

    def append_to_key(self, key, value):
        self.record[key].append(value)

    def get_latest_t(self):
        return self.record["t"][-1]

    def get_latest_acc(self):
        return self.record["acc"][-1]

    def save_record(self, filelabel):
        rows = None
        for arg in self.record.keys():
            new_array = np.expand_dims(np.array(self.record[arg]), axis=1)
            if rows is not None:
                rows = np.concatenate((rows, new_array), axis=1)
            else:
                rows = new_array
        rows = rows.tolist()

        fields = list(self.record.keys())
        filename = filelabel + ".csv"
        with open(filename, 'w') as f:
            write = csv.writer(f)
            write.writerow(fields)
            for row in rows:
                write.writerow(row)

    def save_latest_record(self, filelabel):
        record = []
        for arg in self.record.keys():
            if len(self.record[arg]) > 0:
                record.append(self.record[arg][-1])

        fields = list(self.record.keys())
        filename = filelabel + ".csv"
        with open(filename, 'a+') as f:
            write = csv.writer(f)
            if len(self.record[fields[0]]) == 1:  # Write the fields if this is the first record
                write.writerow(fields)
            write.writerow(record)

    def plot_record(self, figlabel):
        for arg in self.record.keys():
            fig = plt.figure(figsize=(6, 5))
            plt.plot(self.record[arg])
            plt.xlabel('Round #')
            plt.ylabel(arg)
            figname = figlabel + '_{}.png'.format(arg)
            plt.savefig(figname)
            plt.close(fig)


class Profile(object):
    """Clients' loss and delay profile"""
    def __init__(self, num_clients, labels):
        self.loss = np.repeat(-1., num_clients)
        self.delay = np.repeat(-1., num_clients)
        self.primary_label = np.repeat(-1., num_clients)
        self.alpha = 0.1
        self.num_samples = [[]] * num_clients
        self.weights = [[]] * num_clients
        self.grads = [[]] * num_clients
        self.labels = labels

    def set_primary_label(self, pref_str):
        """
        Note, pref is a list of string labels like '3 - three'
        We need to convert the list of string labels to integers
        """
        pref_int = [int(self.labels.index(s)) for s in pref_str]
        self.primary_label = np.array(pref_int)

    def update(self, client_idx, loss, delay, num_samples,
               flatten_weights, flatten_grads):
        if self.loss[client_idx] > 0:
            # Not the first profile
            self.delay[client_idx] = \
                (1 - self.alpha) * self.delay[client_idx] + \
                self.alpha * delay
        else:
            self.delay[client_idx] = delay
        self.num_samples[client_idx] = num_samples
        self.loss[client_idx] = loss
        self.weights[client_idx] = flatten_weights
        self.grads[client_idx] = flatten_grads

    def plot(self, T, path):
        """
        Plot the up-to-date profiles, including loss-delay distribution,
        and 2D PCA plots of normalized weights and grads
        Args:
            T: current time in secs
        """
        def get_cmap(n, name='hsv'):
            '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
            RGB color; the keyword argument name must be a standard mpl colormap name.'''
            return plt.cm.get_cmap(name, n + 1)

        # Plot the loss-delay distribution
        fig = plt.figure()
        cmap = get_cmap(len(set(self.primary_label.tolist())))
        color_ind = 0
        for l in set(self.primary_label.tolist()):
            mask = (self.primary_label == l)
            plt.scatter(x=self.loss[mask], y=self.delay[mask], s=10,
                        color=cmap(color_ind), label=str(l))
            color_ind += 1
        plt.legend()
        plt.xlabel('Loss')
        plt.xlim(left=.0)
        plt.ylabel('Delay (s)')
        plt.ylim(bottom=.0)
        plt.savefig(path + '/ld_{}.png'.format(T))
        plt.close(fig)

        # Plot the PCA of weights
        w_array, l_list = [], []
        for i in range(len(self.weights)):
            if len(self.weights[i]) > 0:  # weight is not empty
                w_array.append(self.weights[i])
                l_list.append(self.primary_label[i])
        w_array, l_array = np.array(w_array), np.array(l_list)
        w_array = StandardScaler().fit_transform(w_array)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(w_array)

        fig = plt.figure()
        cmap = get_cmap(len(list(set(l_list))))
        color_ind = 0
        for l in set(l_list):
            mask = (l_array == l)
            plt.scatter(x=pc[mask, 0], y=pc[mask, 1], alpha=0.8, s=20,
                       color=cmap(color_ind), label=str(l))
            color_ind += 1
        plt.legend()
        plt.title('PCA transform of weights profile')
        plt.savefig(path + '/weight_pca_{}.png'.format(T))
        plt.close(fig)

        # Plot the PCA of grads
        g_array, l_list = [], []
        for i in range(len(self.grads)):
            if len(self.grads[i]) > 0:  # grad is not empty
                g_array.append(self.grads[i])
                l_list.append(self.primary_label[i])
        g_array, l_array = np.array(g_array), np.array(l_list)
        g_array = StandardScaler().fit_transform(g_array)

        pca = PCA(n_components=2)
        pc = pca.fit_transform(g_array)

        fig = plt.figure()
        cmap = get_cmap(len(list(set(l_list))))
        color_ind = 0
        for l in set(l_list):
            mask = (l_array == l)
            plt.scatter(x=pc[mask, 0], y=pc[mask, 1], alpha=0.8, s=20,
                        color=cmap(color_ind), label=str(l))
            color_ind += 1
        plt.legend()
        plt.title('PCA transform of weights profile')
        plt.savefig(path + '/grad_pca_{}.png'.format(T))
        plt.close(fig)

