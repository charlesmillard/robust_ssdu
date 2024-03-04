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


