import argparse
import config
import logging
import os
import time
import server
import random
import numpy as np
import torch
import tensorboard_logger as tb_logger
import glob

# Set up parser
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, default='./config.json',
                    help='Federated learning configuration file.')
parser.add_argument('-sel', '--selection', type=str, default='random',
                    choices=['random', 'high_loss_first', 'short_latency_first',
                             'short_latency_high_loss_first', 'divfl', 'tier', 'oort',
                             'coreset_v1', 'coreset_v2', 'coreset_v3', 'coreset_v4'],
                    help='Client selection algorithm.')
parser.add_argument('-gamma_from', '--cs_gamma_from', type=float, default=0.2,
                    help='Starting weight for delay in client selection.')
parser.add_argument('-gamma_to', '--cs_gamma_to', type=float, default=0.1,
                    help='Finishing weight for delay in client selection')
parser.add_argument('-alpha', '--cs_alpha', type=float, default=1.0,
                    help='Weights for delays')
parser.add_argument('-ass', '--association', type=str, default='random',
                    choices=['random', 'gurobi_v1', 'gurobi_v2'],
                    help='Client association algorithm.')
parser.add_argument('-phi', '--ca_phi', type=float, default=0.1,
                    help='Weight for throughput balancing in client association.')
parser.add_argument('--delay_mode', type=str, default='uniform',
                    choices=['uniform', 'nycmesh'],
                    help='how to generate network delays')
parser.add_argument('--semi_period', type=float, default=70.0,
                    help='Waiting period for semi-async server')
parser.add_argument('--pca_dim', type=int, default=0,
                    help='Dimensions for PCA')
parser.add_argument('--trial', type=int, default=0,
                    help='id for recording multiple runs and setting seeds')
parser.add_argument('-l', '--log', type=str, default='INFO',
                    help='Log messages level.')

args = parser.parse_args()

# Set logging
logging.basicConfig(
    format='[%(levelname)s][%(asctime)s]: %(message)s', level=getattr(logging, args.log.upper()), datefmt='%H:%M:%S')


def main():
    """Run a federated learning simulation."""
    # Set random seed
    random.seed(args.trial)
    np.random.seed(args.trial)
    torch.manual_seed(args.trial)

    # Read configuration file
    fl_config = config.Config(args)

    # Init tensorboard logger
    tb_folder = './tensorboard/' + fl_config.model_name
    if not os.path.isdir(tb_folder):
        os.makedirs(tb_folder)
    logger = tb_logger.Logger(logdir=tb_folder, flush_secs=2)

    # Init model path
    if not os.path.isdir(fl_config.paths.saved_model):
        os.makedirs(fl_config.paths.saved_model)

    # Initialize server
    fl_server = {
        "basic": server.Server(fl_config),
        "accavg": server.AccAvgServer(fl_config),
        "directed": server.DirectedServer(fl_config),
        "kcenter": server.KCenterServer(fl_config),
        "kmeans": server.KMeansServer(fl_config),
        "magavg": server.MagAvgServer(fl_config),
        "dqn": server.DQNServer(fl_config),
        "dqntrain": server.DQNTrainServer(fl_config),
        "sync": server.SyncServer(fl_config),
        "async": server.AsyncServer(fl_config),
        "rflha": server.RflhaServer(fl_config),
        "semiasync": server.SemiAsyncServer(fl_config),
        "pureasync": server.PureAsyncServer(fl_config)
    }[fl_config.server.mode]
    fl_server.boot()

    # Run federated learning
    fl_server.run(logger=logger)

    # Save and plot accuracy-time curve
    #if fl_config.server.mode in ["sync", "async", "semiasync", "rflha", "dqn", "dqntrain"]:
    #    fl_server.records.save_record('{}'.format(fl_config.model_name))
        # fl_server.records.plot_record('{}'.format(fl_config.model_name))

    # Delete global model
    for f in glob.glob(fl_config.paths.model + '/global*'):
        os.remove(f)


if __name__ == "__main__":
    st = time.time()
    main()
    elapsed = time.time() - st
    logging.info('The program takes {} s'.format(
        time.strftime("%Hh%Mm%Ss", time.gmtime(elapsed))
    ))
