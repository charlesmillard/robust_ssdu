# -*- coding: utf-8 -*-
"""
Created on Nov 1st 2021

@author: Charles Millard
"""

import argparse

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from random import sample

from data_loader.zf_data_loader import ZfData
from utils_n2n.preparations import *
from configs.config_loader import *

from trainer.trainer import Trainer

DTYPE = torch.float32


def main(config):
    zf_train = truncate_dataset(ZfData('train', config), config['data']['train_trunc'])
    zf_val = truncate_dataset(ZfData('val', config), config['data']['val_trunc'])

    trainloader = DataLoader(zf_train, batch_size=config['optimizer']['batch_size'], shuffle=True)
    validloader = DataLoader(zf_val, batch_size=3 * config['optimizer']['batch_size'])

    # create log directory
    logdir = config['network']['save_loc']
    if ~os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)
    print('Saving results to directory ', logdir)

    # save config file
    with open(logdir + '/config', 'w') as fp:
        yaml.dump(config, fp, default_flow_style=False)

    pass_network, network = create_network(config)
    optimizer = create_optimizer(network, config)

    if config['optimizer']['sched_mstones'] is not None:
        scheduler = MultiStepLR(optimizer, milestones=config['optimizer']['sched_mstones'], gamma=0.1)
    else:
        scheduler = None

    criterion = create_criterion(config)
    if config['optimizer']['load_model_root'] is not None:
        network = load_parameters(network, config)

    trainer = Trainer(network, pass_network, criterion, optimizer,
                      scheduler, config, trainloader, validloader)

    trainer.train()


if __name__ == '__main__':
    torch.set_default_dtype(DTYPE)
    torch.backends.cudnn.enabled = False

    #test

    args = argparse.ArgumentParser(description='Robust SSDU')
    args.add_argument('-c', '--config', default='default.yaml', type=str,
                      help='config file path (default: default.yaml)')
    args.add_argument('-l', '--log_loc', default='saved/logs/cpu/default/', type=str,
                      help='file path to save logs (default: saved/logs/cpu/default/)')
    args.add_argument('-d', '--data_loc', default='/home/xsd618/data/fastMRI_test_subset_brain/', type=str,
                      help='data location (default: /home/xsd618/data/fastMRI_test_subset_brain/)')

    args = args.parse_args()

    # load and setup config
    config_main = load_config('configs/' + args.config)
    config_main = reformat_config(config_main)
    config_main = set_missing_config_entries(config_main)
    config_main = prepare_config(config_main, args)

    print('using config file {}'.format(args.config))

    set_seeds(config_main['optimizer']['seed'])

    main(config_main)
