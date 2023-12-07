import yaml
from utils_n2n.utils import *


def load_config(cname):
    """ loads yaml file """
    with open(cname, 'r') as stream:
        configs = yaml.safe_load_all(stream)
        config = next(configs)
    return config


def set_missing_config_entries(config):
    default_config = load_config('configs/default.yaml')

    # set elements in dictionary to default if missing
    for level1 in default_config:
        if level1 in config:
            for level2 in default_config[level1]:
                if level2 not in config[level1]:
                    config[level1][level2] = default_config[level1][level2]
                    warnings.warn('Dictionary element {} from {} missing from config file. Set as default {}'
                                  .format(level2, level1, config[level1][level2]))
        else:
            config[level1] = default_config[level1]
            warnings.warn('Dictionary element {} missing from config file. Set as default {}'
                          .format(level1, config[level1]))

    return config


def prepare_config(config, args):
    if torch.cuda.is_available():
        config["network"]["device"] = 'cuda'
    else:
        config["network"]["device"] = 'cpu'
        # config['optimizer']['batch_size'] = 2  # small batch size so can handle on my machine
        # config['network']['ncascades'] = 2

    config['network']['save_loc'] = args.log_loc
    config['data']['loc'] = args.data_loc

    return config


def reformat_config(config):
    if 'method' in config['data']:
        config['optimizer']['method'] = config['data']['method']
        config['mask'] = {'sample_type': config['data']['sample_type'], 'sample_type_lambda': config['data']['sample_type'],
                          'fully_samp_size': config['data']['fully_samp_size'], 'fully_samp_size_lambda': config['data']['fully_samp_size'],
                          'poly_order': config['data']['poly_order'], 'poly_order_lambda': config['data']['poly_order'],
                          'us_fac': config['data']['us_fac'], 'us_fac_lambda': config['data']['us_fac_lambda']}

        if 'sim_noise' not in config['data']:
            config['data']['sim_noise'] = True

        config['noise'] = {'sim_noise': config['data']['sim_noise'],
                           'sigma1': config['data']['sigma1'],
                           'alpha': config['data']['sigma2'] / config['data']['sigma1'] }

        config['optimizer']['load_model_root'] = None
        config['optimizer']['sched_mstones'] = None
        config['optimizer']['noise2recon_lamb'] = 1

        config['data']['fixed_ncoil'] = 16

        if config['network']['denoi_model'] == 'unet_single':
            config['network']['denoi_model'] = 'unet'
        elif config['network']['denoi_model'] == 'unet':
            config['network']['denoi_model'] = 'split_unet'

        config['data']['set'] = "fastmri"
        config['data']['loc'] = data_root(config)

    if config['optimizer']['method'] == "n2n_ssdu":
        config['optimizer']['method'] = "robust_ssdu"
    elif config['optimizer']['method'] == "n2n":
        config['optimizer']['method'] = "noisier2full"
    elif config['optimizer']['method'] == "full_noisy":
        config['optimizer']['method'] = "noise2full"

    # if config['data']['loc'][-1] == "_":
    #     config['data']['loc'] = config['data']['loc'][:-10]

    return config
