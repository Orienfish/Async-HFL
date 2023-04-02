from collections import namedtuple
import json

class Config(object):
    """Configuration module."""

    def __init__(self, args):
        self.paths = ""
        # Load config file
        with open(args.config, 'r') as config:
            self.config = json.load(config)
        self.selection = args.selection
        self.cs_gamma_from = args.cs_gamma_from
        self.cs_gamma_to = args.cs_gamma_to
        self.cs_alpha = args.cs_alpha
        self.association = args.association
        self.ca_phi = args.ca_phi
        self.delay_mode = args.delay_mode
        self.semi_period = args.semi_period
        self.pca_dim = args.pca_dim
        self.trial = args.trial

        # Extract configuration
        self.extract()

    def extract(self):
        config = self.config

        # -- Model --
        self.model = config['model']

        # -- Clients --
        fields = ['total', 'per_round', 'label_distribution',
                  'do_test', 'test_partition']
        defaults = (0, 0, 'uniform', False, 0.2)
        params = [config['clients'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.clients = namedtuple('clients', fields)(*params)

        assert self.clients.per_round <= self.clients.total

        # -- Data --
        fields = ['loading', 'partition', 'IID', 'bias', 'shard', 'noniid']
        defaults = ('static', 0, False, None, None, None)
        params = [config['data'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.data = namedtuple('data', fields)(*params)

        # Determine correct data loader
        if self.model in ['MNIST', 'FashionMNIST', 'CIFAR-10']:
            assert self.data.IID ^ bool(self.data.bias) ^ \
                   bool(self.data.shard) ^ bool(self.data.noniid)
            if self.data.IID:
                self.loader = 'basic'
            elif self.data.bias:
                self.loader = 'bias'
            elif self.data.shard:
                self.loader = 'shard'
            elif self.data.noniid:
                self.loader = 'noniid'
        else:
            self.loader = 'leaf'

        # -- Federated learning --
        fields = ['target_accuracy', 'task', 'epochs', 'batch_size',
                  'model_size']
        defaults = (None, 'train', 0, 0, 0)
        params = [config['federated_learning'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.fl = namedtuple('fl', fields)(*params)

        # -- Server --
        fields = ['mode', 'rounds', 'adjust_round']
        defaults = ('sync', 400, 20)
        params = [config['server'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.server = namedtuple('server', fields)(*params)

        # -- Gateways --
        fields = ['mode', 'rounds', 'total', 'throughput_ub']
        defaults = ('sync', 5, 1, 1000)
        params = [config['gateways'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.gateways = namedtuple('gateways', fields)(*params)

        # -- Paths --
        fields = ['data', 'model', 'saved_model', 'reports', 'plot']
        defaults = ('./data', './models', None, None, './plots')
        params = [config['paths'].get(field, defaults[i])
                  for i, field in enumerate(fields)]

        # Set specific model path
        if self.loader == 'leaf' or self.data.IID:  # IID
            distrib = 'na'
        elif self.data.bias:  # bias loader
            distrib = 'p{}s{}'.format(
                float(self.data.bias['primary']), float(self.data.bias['secondary'])
            )
        elif self.data.noniid:  # nonIID loader
            distrib = 'min{}max{}'.format(
                int(self.data.noniid['min_cls']), int(self.data.noniid['max_cls'])
            )
        else:
            raise ValueError("data distribution type not implemented")

        self.model_name = '{}_{}_{}_iid{}_{}_c{}_th{}_{}_{}_{}_{}_{}_{}_{}'.format(
            self.model, self.server.mode, self.delay_mode, int(self.data.IID),
            distrib, self.clients.total, self.gateways.throughput_ub,
            self.selection, self.cs_alpha,
            self.association, self.ca_phi,
            self.semi_period, self.pca_dim, self.trial
        )

        params[fields.index('model')] += '/' + self.model
        params[fields.index('saved_model')] = params[fields.index('model')] + \
                                              '/' + self.model_name

        self.paths = namedtuple('paths', fields)(*params)

        # -- Async --
        fields = ['alpha', 'gl_alpha', 'rou', 'staleness_func', 'llambda']
        defaults = (0.4, 0.9, 1.0, 'constant', 0.5)
        params = [config['async'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.async_params = namedtuple('async_params', fields)(*params)

        # -- Link Speed --
        fields = ['min', 'max', 'std', 'sparse_ratio']
        defaults = (200, 5000, 100, 0.5)
        params = [config['link_speed'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.link = namedtuple('link_speed', fields)(*params)

        # -- Computational Time --
        fields = ['min', 'max', 'std']
        defaults = (15, 100, 10)
        params = [config['comp_time'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.comp_time = namedtuple('comp_time', fields)(*params)

        # -- Delays --
        fields = ['cloud_gateway', 'gateway_client', 'comp_time']
        defaults = (0, 0, 0)
        params = [config['delays'].get(field, defaults[i])
                  for i, field in enumerate(fields)]
        self.delays = namedtuple('delays', fields)(*params)

