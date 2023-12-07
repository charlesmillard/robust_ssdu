import os

from utils.preparations import *


config_name = 'stem.yaml'  # + '.yaml'
config = load_config(config_name)

save_root = 'other_losses/'

allR = [8]
sigma1 = [8e-2]
# sigma1 = [1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
method = ['full', 'full_noisy', 'ssdu', 'n2n', 'n2n_ssdu', 'noise2recon']
# method = ['rei']

for s1 in sigma1:
    for R in allR:
        for meth in method:
            config['data']['sigma1'] = s1
            config['data']['sigma2'] = s1/2
            config['data']['us_fac'] = R
            config['data']['method'] = meth

            if meth in ['n2n_ssdu', 'n2n']:
                config['optimizer']['alpha_weight'] = True
                config['optimizer']['K_weight'] = True
            else:
                config['optimizer']['alpha_weight'] = False
                config['optimizer']['K_weight'] = False

            config['optimizer']['batch_size'] = 1 if meth == 'rei' else 2

            logdir = save_root + str(R) + 'x/' + str(s1)

            if ~os.path.isdir(logdir):
                os.makedirs(logdir, exist_ok=True)

            with open(logdir + '/' + meth + '.yaml', 'w') as fp:
                yaml.dump(config, fp, default_flow_style=False)

print('total number of configs is {}'.format(len(allR) * len(sigma1) * len(method)))
