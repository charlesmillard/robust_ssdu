from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from utils.preparations import *

import os

sd = 580 # 260,  320, 420, 270. 570, 580
show_main_res = False

# type = "mag_comp" #"n2n_vs_sure_ssdu"
# if type == 'weight_test':
#     root = 'logs/cuda/'
#     # log_loc = [root + '75818101', root + '76024129' ]
#     sig = '0.04'
#     log_loc = [root + 'full.yaml/15600247',
#                root + 'n2n_ssdu_no_weight.yaml/15600913',
#                root + 'n2n_ssdu.yaml/15457325']
# elif type == 'proper_mse':
#     root = 'logs/cuda/4x/'
#     # log_loc = [root + '75818101', root + '76024129' ]
#     sig = '0.04'
#     log_loc = [root + 'full.yaml/16120026',
#                root + 'full.yaml/16392291',
#                root + 'n2n_ssdu_no_weight.yaml/16392290',
#                root + 'n2n_ssdu.yaml/16392292']
# elif type == 'weight_test_4':
#     root = 'logs/cuda/4x/'
#     # log_loc = [root + '75818101', root + '76024129' ]
#     sig = '0.04'
#     log_loc = [root + 'full.yaml/16120026',
#                root + 'n2n_ssdu_no_weight.yaml/16392290 ',
#                root + 'n2n_ssdu.yaml/16120027']
# if type == 'test':
#     root = 'logs/cuda/'
#     log_loc = [root + 'n2n_ssdu_im_mag.yaml/16815312']
# elif type == 'mag_comp':
#     root = 'logs/cuda/'
#     log_loc = [root + 'whole_experiment/8x/0.02/ssdu.yaml/16462407',
#                root + 'n2n_ssdu_im_mag.yaml/16815312']
#
# elif type == "8x_meth_test":
#     root = 'logs/cuda/whole_experiment/'
#     sig = '0.04'
#     log_loc = [root + 'full/76846908_' + sig, root + 'full_noisy/76846908_' + sig,
#                root + 'n2n/76846908_' + sig, root + 'ssdu/76846908_' + sig, root + 'n2n_ssdu/76846908_' + sig]
# elif type == "4x_meth_test":
#     root = 'logs/cuda/4x/'
#     sig = '0.04'
#     log_loc = [root + 'full/76846908_' + sig, root + 'full_noisy/76846908_' + sig,
#                root + 'n2n/76846908_' + sig, root + 'ssdu/76846908_' + sig, root + 'n2n_ssdu/76846908_' + sig]
# elif type == "4v8":
#     root = 'logs/cuda/'
#     sig = '0.0'
#     log_loc = [root + '4x/full/76846908_' + sig, root  + '8x/full/76846908_' + sig]
# elif type == "deg":
#     root = 'logs/cuda/4x/'
#     log_loc = [root + 'ssdu/76846908_0.02', root + 'ssdu/76846908_0.04',
#                root + 'ssdu/76846908_0.06', root + 'ssdu/76846908_0.08']
# elif type == "no_deg":
#     root = 'logs/cuda/4x/'
#     log_loc = [root + 'n2n_ssdu/76846908_0.02', root + 'n2n_ssdu/76846908_0.04',
#                root + 'n2n_ssdu/76846908_0.06', root + 'n2n_ssdu/76846908_0.08']
# elif type == "no_deg_full":
#     root = 'logs/cuda/4x/'
#     log_loc = [root + 'full/76846908_0.02', root + 'full/76846908_0.04',
#                root + 'full/76846908_0.06', root + 'full/76846908_0.08']
# elif type == "rei":
#     log_loc = ['logs/cuda/8x/n2n_ssdu/76846908_0.04', 'logs/cuda/rei_config.yaml/12424082']
#     # log_loc = ['logs/cuda/4x/n2n_ssdu/76846908_0.04', 'logs/cuda/81077585']
# elif type == "n2n_vs_sure_ssdu":
#     # log_loc = ['logs/cuda/8x/n2n_ssdu/76846908_0.04', 'logs/cuda/sure_ssdu.yaml/81196597'] #, 'logs/cuda/sure_ssdu/81055149']
#     log_loc = ['logs/cuda/denoising_proxy_test/n2n_ssdu2.yaml/81196193',  'logs/cuda/sure_ssdu.yaml/81196597']
# elif type == "n2n_ssdu_autotune":
#     log_loc = ['logs/cuda/denoising_proxy_test/n2n_ssdu2.yaml/81196193',
#                'logs/cuda/denoising_proxy_test/n2n_ssdu6.yaml/81196193']
# elif type == "n2n_sure_rei_noise2recon":
#     # log_loc = ['logs/cuda/8x/n2n_ssdu/76846908_0.04', 'logs/cuda/sure_ssdu.yaml/81196597', 'logs/cuda/81152985']
#     log_loc = ['logs/cuda/denoising_proxy_test/n2n_ssdu2.yaml/81196193',
#                'logs/cuda/sure_ssdu.yaml/81213568', 'logs/cuda/rei_config.yaml/81203370',
#                'logs/cuda/noise2recon.yaml/12424730']
# elif type == "noise2recon":
#     log_loc = ['logs/cuda/denoising_proxy_test/n2n_ssdu2.yaml/81196193', 'logs/cuda/noise2recon.yaml/12424730']

type = 'ss/4x_0.06/'
accels = [4]
sigmas = [0.06]

roots = ['/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment_alph1',
        '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment_alph1_weighted',
         '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment',
        '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/weighted_experiment',
         ]

# meths = ['full', 'full_noisy', 'n2n', 'n2n', 'ssdu', 'noise2recon', 'rei',  'n2n_ssdu',  'n2n_ssdu']
# is_weighted = [None, None, False, True, None, None, None, False,  True]

meths = ['full', 'full_noisy', 'n2n', 'n2n', 'ssdu', 'noise2recon',  'n2n_ssdu',  'n2n_ssdu']
is_weighted = [None, None, False, True, None,  None, False,  True]

#subset = (0, 1, 2, 3, 4, 5, 6, 7) # everything
#subset = (0, 1, 2, 3) # noisy, fully sampled training data
subset = (0, 4, 5, 6, 7) # noisy, sub-sampled sampled training data

# subset = (0,)

meths = [meths[i] for i in subset]
is_weighted = [is_weighted[i] for i in subset]

n_accels = len(accels)
n_sigmas = len(sigmas)
n_meths = len(meths)

all_logs = []
for r in roots:
    for f in os.walk(r):
        if 'state_dict' in f[2]:
            all_logs.append(f[0])

log_loc = np.zeros((n_accels, n_sigmas, n_meths), dtype=object)

for log in all_logs:
    config = load_config(log + '/config')
    ac = config['data']['us_fac']
    sig = config['data']['sigma1']
    meth = config['data']['method']
    k_weighted = config['optimizer']['K_weight']

    ac_idx = accels.index(ac) if ac in accels else 'none'
    sig_idx = sigmas.index(sig) if sig in sigmas else 'none'

    meth_idx = 'none'
    for ii in range(len(meths)):
        if meths[ii] == meth and is_weighted[ii] in (k_weighted, None):
            meth_idx = ii

    if 'none' not in [ac_idx, sig_idx, meth_idx]:
        print(log)
        if log_loc[ac_idx, sig_idx, meth_idx] == 0:
            log_loc[ac_idx, sig_idx, meth_idx] = log
        else:
            print('double attempted at ({}, {}, {})'.format(ac_idx, sig_idx, meth_idx))


log_loc = np.reshape(log_loc, (n_accels*n_sigmas*n_meths))

xz = 120
yz = 60
wth = 100

all_nmse = []

with torch.no_grad():
    for l in log_loc:
        print('log directory: ' + l)

        if show_main_res:
            res = np.load(l + '/results.npz')
            print('Mean over test set is {}'.format(np.mean(res['loss'])))

        config = load_config(l + '/config')
        config['network']['device'] = 'cpu'
        meth = config['data']['method']
        print('Method is: ' + meth)
        print('Accel is: ' + str(config['data']['us_fac']))

        pass_network, network = create_network(config)
        optimizer = create_optimizer(network, config)
        criterion = create_criterion(config)

        sigma1 = float(config['data']['sigma1'])
        sigma2 = float(config['data']['sigma2'])
        if sigma1 > 0:
            alpha_sq = (sigma2 / sigma1) ** 2
        # alpha_sq = 1
        print('sigma1 is: ' + str(sigma1))

        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=True)
        network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
        network.to(config['network']['device'])
        network.eval()
        for i, data in enumerate(test_load, 0):
            y0, noise1, noise2, mask_omega, mask_lambda, Kweight, mask = data
            # print(torch.mean(mask_omega.float()))
            # mu = mu[0] if meth == "n2n_ssdu" else 1  # all the same
            noise1 *= sigma1
            noise2 *= sigma2

            # noise1 = 0

            y = mask_omega * (y0 + noise1)

            below_noise_fl = torch.as_tensor(torch.abs(y0[0, 0, 0]) < sigma1, dtype=float)

            print(meth, sigma1)

            pad = torch.abs(y0) == 0

            if meth in ["n2n_ssdu"]:
                y_tilde = y
                # y_tilde = mask_lambda*y
                # y_tilde = mask_lambda*(y + mask_omega * noise2)
                outputs = pass_network(y_tilde, network)
                y0_est = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)

            elif meth in ["n2n"]:
                y_tilde = y
                #y_tilde = y + mask_omega * noise2
                outputs = pass_network(y_tilde, network)
                y0_est = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)

            elif meth in ["ssdu", "ssdu_bern", "sure_ssdu"]:
                #y_tilde = mask_lambda * y  # second mask
                y_tilde = y
                # outputs = pass_network(y_tilde, network)
                # y0_est = outputs * (y == 0) + y
                y0_est = pass_network(y_tilde, network)
            elif meth in ["full", "full_noisy", "noise2recon", "rei"]:
                outputs = pass_network(y, network)
                y0_est = outputs

            y0_est[pad] = 0  # set y0 padding to zero
            x0_est = kspace_to_rss(y0_est)
            x0 = kspace_to_rss(y0)

            pad_mask = x0 > 0.0

            print('MSE: {:e}'.format(mse_loss(y0_est, y0)))
            all_nmse.append(nmse_loss(y0_est, y0).item())

            from utils.mask_tools import gen_pdf
            prob_omega = gen_pdf(config['data']['nx'], config['data']['ny'], 1 / config['data']['us_fac'], config['data']['poly_order'],
                                 config['data']['fully_samp_size'], config['data']['sample_type'])
            prob_omega = torch.as_tensor(prob_omega.copy(), dtype=torch.float)
            # approx_NMSE = nmse_loss(mask_omega * y0_est / prob_omega,  mask_omega * y0 / prob_omega)
            # approx_MSE = (prob_omega**(-1) - 1) * mask_omega * (y0_est - y)**2 / prob_omega  # + mask_omega * sigma1**2 / prob_omega
            approx_MSE =  mask_omega * (y0_est - y) ** 2 / prob_omega # + mask_omega  *  sigma1**2 / prob_omega
            print('Est MSE: {:e}'.format(torch.mean(approx_MSE)))

            net = lambda y: (mask_omega / prob_omega) * ((1 + alpha_sq) * pass_varnet(y, network) - y) / alpha_sq
            sure_MSE_est = sure_loss(net, y / prob_omega, y0_est / prob_omega, mask_omega, 1e-3, torch.as_tensor(sigma1))
            print('Est MSE, SURE: {:e}'.format(torch.mean(sure_MSE_est)))

            break

        break



