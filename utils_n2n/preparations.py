from utils_n2n.utils import *
from fastmri.models.unet import Unet
from model.varnet_denoi_recon_split import VarNet

from random import sample


def create_criterion(config):
    loss_choice = config['optimizer']['loss']
    if loss_choice == 'mse':
        criterion = mse_loss
    elif loss_choice == 'nmse':
        criterion = nmse_loss
    elif loss_choice == 'rmse':
        criterion = rmse_loss
    elif loss_choice == 'l1':
        criterion = torch.nn.L1Loss()
    elif loss_choice == 'l1l2':
        criterion = l12loss
    elif loss_choice == 'im_mag':
        criterion = image_mag_loss
    else:
        raise NameError('You have chosen an invalid loss')
    return criterion


def load_parameters(base_net, config):
    loc = str(config['optimizer']['load_model_root'])
    base_net.load_state_dict(torch.load(loc + '/state_dict', map_location=config["network"]["device"]))
    print('model loading successful, using ' + str(config['optimizer'][
                                                       'load_model_root']) + ' as parameter initialisation')
    return base_net


def create_network(config):
    if config['network']['type'] == 'unet':
        chans = 2 * config['data']['fixed_ncoil']
        base_net = Unet(in_chans=chans, out_chans=chans).to(config['network']['device'])
        pass_network = pass_unet
    elif config['network']['type'] == 'varnet':
        base_net = VarNet(num_cascades=config['network']['ncascades'],
                          denoi_model=config['network']['denoi_model']).to(config['network']['device'])
        pass_network = pass_varnet
    else:
        raise Exception('Invalid network type chosen')

    pytorch_total_params = compute_number_of_params(base_net)
    print('Network type {} created with {:e} parameters'.format(config['network']['type'], pytorch_total_params))
    return pass_network, base_net


def create_optimizer(base_net, config):
    if config['optimizer']['name'] == 'Adam':
        optimizer = torch.optim.Adam(base_net.parameters(), lr=float(config['optimizer']['lr'])
                                     , weight_decay=float(config['optimizer']['weight_decay']))
    elif config['optimizer']['name'] == 'SGD':
        optimizer = torch.optim.SGD(base_net.parameters(), lr=float(config['optimizer']['lr'])
                                    , weight_decay=float(config['optimizer']['weight_decay']),
                                    momentum=float(config['optimizer']['momentum']))
    else:
        raise NameError('You have chosen an invalid optimizer name')

    return optimizer


def truncate_dataset(dataset, trunc):
    n_data = dataset.__len__()
    if trunc is not None:
        if n_data > trunc:
            print('{} dataset truncated from {} slices to {}'.format(dataset.__name__, n_data, trunc))
            dataset = torch.utils.data.Subset(dataset, sample(range(n_data), trunc))
        else:
            print('{} Dataset has {} slices and requested truncation is {}, so ignored'.format(dataset.__name__, n_data, trunc))
    else:
        print('{} dataset contains {} slices'.format(dataset.__name__, n_data))

    return dataset
