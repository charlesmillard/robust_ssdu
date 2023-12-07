import torch.cuda
import torchvision

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from data_loader.zf_data_loader import ZfData
from utils.preparations import *

import os
import itertools

sd = 320 # 260,  320, 420, 270. 570, 580
show_main_res = False

file_choice = 'file_brain_AXFLAIR_200_6002533.h5' # 'file_brain_AXFLAIR_200_6002548.h5'
file_choice = '2022062708_T102.h5'
slice_choice = 6
is_lowfield = True

use_slice_choice = True

accels = [4]
sigmas = [0.02]

type = '/home/xsd618/noisier2noise_kspace_denoising/saved_images/alpha_comp/' + str(accels[0]) + 'x_' + str(sigmas[0])

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
#subset = (0, 4, 5, 6, 7) # noisy, sub-sampled sampled training data
subset = (7, )
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


# log_loc = [ 'logs/cuda/weighted_experiment/8x/0.08/n2n_ssdu.yaml/17637340',
#             'logs/cuda/other_losses/8x/0.08/n2n_ssdu.yaml/21570489']

# log_loc = ['logs/cuda/whole_experiment/8x/0.06/n2n_ssdu.yaml/16421958',
#            'logs/cuda/weighted_experiment/8x/0.06/n2n_ssdu.yaml/17584179',
#            'logs/cuda/weighted_experiment/8x/0.06/n2n_ssdu_no_alpha.yaml/20383080',
#            'logs/cuda/weighted_experiment/8x/0.06/n2n_ssdu_no_K.yaml/20383091']

# log_loc = ['logs/cuda/whole_experiment/4x/0.08/full.yaml/20649536',
#            'logs/cuda/whole_experiment/4x/0.08/full.yaml/17481871']
#
# log_loc = ["logs/cuda/whole_experiment/8x/0.08/full.yaml/16791123",
#            "logs/cuda/lr_scheduling/full.yaml/22691526",
#            'logs/cuda/weighted_experiment/8x/0.08/n2n_ssdu.yaml/17637340',
#            "logs/cuda/lr_scheduling/n2n_ssdu.yaml/22772950"]

# log_loc = ['logs/cuda/alpha_robustness/0.25/noise2recon.yaml/20391310',
#            'logs/cuda/whole_experiment/8x/0.06/noise2recon.yaml/16422720',
#            'logs/cuda/alpha_robustness/0.75/noise2recon.yaml/20385429',
#            'logs/cuda/alpha_robustness/1/noise2recon.yaml/20391645',
#            "logs/cuda/alpha_robustness/1.25/noise2recon.yaml/22549540",
#            "logs/cuda/alpha_robustness/1.75/noise2recon.yaml/22553770",
#            "logs/cuda/alpha_robustness/2/noise2recon.yaml/22462199"]
#
# log_loc = ['logs/cuda/whole_experiment/8x/0.06/noise2recon.yaml/16422720',
#            'logs/cuda/weighted_experiment/8x/0.06/noise2recon.yaml/17614579',
#            'logs/cuda/alpha_robustness/1/noise2recon.yaml/20391645',
#            "logs/cuda/alpha_robustness/1.25/n2n_ssdu_weighted.yaml/22518839"]

# log_loc = ['logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#            'logs/cuda/alpha_robustness/0.25/n2n_ssdu_weighted.yaml/20391283',
#            'logs/cuda/weighted_experiment/8x/0.06/noise2recon.yaml/17614579',
#            'logs/cuda/alpha_robustness/0.75/n2n_ssdu_weighted.yaml/20396269',
#            'logs/cuda/alpha_robustness/1/n2n_ssdu_weighted.yaml/20391458',
#            "logs/cuda/alpha_robustness/1.25/n2n_ssdu_weighted.yaml/22518839",
#            "logs/cuda/alpha_robustness/1.5/n2n_ssdu_weighted.yaml/22591989",
#            "logs/cuda/alpha_robustness/1.75/n2n_ssdu_weighted.yaml/22549561",
#            "logs/cuda/alpha_robustness/2/noise2recon.yaml/22462199"]

# log_loc = ["logs/cuda/whole_experiment/8x/0.08/full.yaml/16791123",
#            "logs/cuda/whole_experiment/8x/0.08/n2n_ssdu.yaml/16761481",
#            "logs/cuda/whole_experiment_alph1/8x/0.08/n2n_ssdu.yaml/23064460",
#            "logs/cuda/whole_experiment_alph1_weighted/8x/0.08/n2n_ssdu.yaml/23315191"]

# log_loc = ['logs/cuda/whole_experiment/4x/0.06/n2n_ssdu.yaml/17122611',
#            'logs/cuda/whole_experiment_alph1/4x/0.06/n2n_ssdu.yaml/23091533']
#
# log_loc = ['logs/cuda/whole_experiment/4x/0.08/ssdu.yaml/17177269',
#            'logs/cuda/whole_experiment/4x/0.08/noise2recon.yaml/17792587',
#            'logs/cuda/whole_experiment_alph1/4x/0.08/noise2recon.yaml/23213976']
#
# log_loc = ['logs/cuda/weighted_experiment/8x/0.06/n2n_ssdu.yaml/17584179',
#             'logs/cuda/whole_experiment_alph1_weighted/8x/0.06/n2n_ssdu.yaml/23685871']
#
# log_loc = ['logs/cuda/weighted_experiment/4x/0.04/n2n_ssdu.yaml/17737025',
#             'logs/cuda/whole_experiment_alph1_weighted/4x/0.04/n2n_ssdu.yaml/23638603']
# #
# # log_loc = ['logs/cuda/whole_experiment/8x/0.08/full.yaml/16791123',
# #            'logs/cuda/architectures/8x/0.08/full.yaml/24282339']
#
# log_loc = ['logs/cuda/whole_experiment/8x/0.08/full.yaml/16791123',
#           'logs/cuda/whole_experiment_alph1_weighted/8x/0.08/n2n_ssdu.yaml/23685816']


# log_loc = ['logs/cuda/whole_experiment/8x/0.08/full.yaml/16791123',
#             'logs/cuda/other_losses/8x/0.08/full.yaml/24461337',
#            'logs/cuda/other_losses/8x/0.08/n2n_ssdu.yaml/24563985']
#
# log_loc = ['logs/cuda/other_losses/8x/0.08/full.yaml/24461337',
#            'logs/cuda/other_losses/8x/0.08/n2n_ssdu.yaml/24563985']

# log_loc = ['logs/cuda/whole_experiment/8x/0.04/n2n_ssdu.yaml/16791128',
#            'logs/cuda/bernoulli_samp/8x/0.04/n2n_ssdu.yaml/24647038',
#            'logs/cuda/bernoulli_samp/8x/0.04/full.yaml/24661889']

xz = 120
yz = 60
x_wth = 100
y_wth = 100

# xz = 148
# yz = 178
# x_wth = 17
# y_wth = 25

# xz = 83
# yz = 106
# x_wth = 23
# y_wth = 40

all_ssim_hat = []
all_nmse = []
correct_i = 'None'

alphas = [1.5, 2, 2.5, 100]

with torch.no_grad():
    for alp in alphas:
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

            alpha_sq = alp ** 2
            # alpha_sq = 1
            print('alp is: ' + str(alp))

            np.random.seed(sd)
            torch.manual_seed(sd)
            torch.cuda.manual_seed_all(sd)

            test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=False)

            network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
            network.to(config['network']['device'])
            network.eval()

            if use_slice_choice:
                if correct_i == 'None':
                    for i, data in enumerate(test_load, 0):
                        slice_info = data[-1]
                        print(slice_info)
                        is_correct_slice = (slice_info[0][0] == file_choice and int(slice_info[1]) == slice_choice)
                        #print(is_correct_slice)
                        if is_correct_slice:
                            correct_i = i
                            break
            else:
                correct_i = np.random.randint(254)

            y0, noise1, noise2, mask_omega, mask_lambda, Kweight, mask, slice_info = \
                next(itertools.islice(test_load, correct_i, None))
            # y0, noise1, noise2, mask_omega, mask_lambda, Kweight, mask, slice_info = data
            # print(torch.mean(mask_omega.float()))
            # mu = mu[0] if meth == "n2n_ssdu" else 1  # all the same

            noise1 *= sigma1
            noise2 *= sigma2

            if is_lowfield:
                noise1 = 0
                y0[:, :, :, :, :40] = 0
                y0[:, :, :, :, -40:] = 0

            y = mask_omega * (y0 + noise1)

            below_noise_fl = torch.as_tensor(torch.abs(y0[0, 0, 0]) < sigma1, dtype=float)

            print(meth, sigma1)

            pad = torch.abs(y0) == 0
            y[pad] = 0

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

            x0_est = x0_est * pad_mask
            x0 = x0 * pad_mask

            print('Est NMSE: {:e}'.format(nmse_loss(y0_est, y0)))
            print('Est MSE: {:e}'.format(torch.mean((y0_est - y0)**2)))
            all_nmse.append(nmse_loss(y0_est, y0).item())

            print('MSE on sampled: {:e}'.format(torch.mean(mask_omega * mask_lambda * (y0_est - y0) ** 2)))
            print('MSE on unsampled: {:e}'.format(torch.mean((1 - mask_omega * mask_lambda) * (y0_est - y0) ** 2)))

            from utils.mask_tools import gen_pdf
            prob_omega = gen_pdf(config['data']['nx'], config['data']['ny'], 1 / config['data']['us_fac'], config['data']['poly_order'],
                                 config['data']['fully_samp_size'], config['data']['sample_type'])
            prob_omega = torch.as_tensor(prob_omega.copy())
            # approx_NMSE = nmse_loss(mask_omega * y0_est / prob_omega,  mask_omega * y0 / prob_omega)
            alpha_weight = np.sqrt((1 + alpha_sq) / alpha_sq)
            #approx_MSE_sampled = (Kweight * mask_omega * (1 - mask_lambda)* (y0_est - y))**2
            approx_MSE_sampled = (mask_omega * (1 - mask_lambda) * (y0_est - y) / prob_omega**0.5 ) ** 2
            approx_MSE_unsampled = (alpha_weight * mask_omega * mask_lambda * (y0_est - y))**2
            approx_MSE = approx_MSE_sampled + approx_MSE_unsampled

            # approx_MSE = (prob_omega**(-1) - 1) * mask_omega * (y0_est - y)**2 / prob_omega  # + mask_omega * sigma1**2 / prob_omega
            # approx_MSE =  mask_omega * (y0_est - y) ** 2 / prob_omega # + mask_omega  *  sigma1**2 / prob_omega
            print('Est MSE approx: {:e}'.format(torch.mean(approx_MSE)))

            print('MSE est on sampled: {:e}'.format(torch.mean(approx_MSE_sampled)))
            print('MSE est on unsampled: {:e}'.format(torch.mean(approx_MSE_unsampled)))


            print('Est NMSE (image domain): {:e}'.format(mse_loss(x0_est, x0)))

            l1l2_loss = mse_loss(mask_omega * y0_est, mask_omega * y0) + \
                        l12loss((1 - mask_omega) * y0_est, (1 - mask_omega) * y0)
            print('Est l1-l2: {:e}'.format(l1l2_loss))

            print('Est NMSE (image domain): {:e}'.format(mse_loss(x0_est, x0)))
            print('Est NMSE (image domain) (dB): {:e}'.format(- 10 * torch.log10(torch.mean((x0_est-x0)**2))))

            mx = torch.max(torch.abs(x0[0]))

            x_input = kspace_to_rss(y)
            y_ssdu = (y != 0) * y + y0_est * (y == 0)
            y_ssdu[pad] = 0
            x_ssdu = kspace_to_rss(y_ssdu)

            # x_err = kspace_to_rss(y0_est - y0)
            # input_err = kspace_to_rss(y - y0)

            x_err = torch.abs(x0_est - x0)
            input_err = torch.abs(x_input - x0)

            print('Input NMSE (image_domain): {:e}'.format(nmse_loss(x_input, x0)))
            print('Input NMSE (k-space): {:e}'.format(nmse_loss(y, y0)))

            nx = y.shape[-2]
            if torch.numel(mask < 10):
                mask = kspace_to_im(y0) < 0.2

            m = mask[0, 0, 0, nx // 2 - 160:nx // 2 + 160]

            x_im = kspace_to_im(y0_est)
            masked_im = mask * x_im
            masked_im = masked_im[masked_im != 0]
            print('Estimate of noise std is {}'.format(torch.std(masked_im)))

            x2 = np.array((m * x0[0, 0]).detach().cpu())
            x1 = np.array((m * x0_est[0, 0]).detach().cpu())
            all_ssim_hat.append(ssim(x1, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
            # all_ssim_hat.append(ssim(x1[xz:xz + x_wth, yz:yz + y_wth], x2[xz:xz + x_wth, yz:yz + y_wth],
            #                          data_range=x2.max()))

            if 'x0_est_all' in locals():
                x0_est_all = torch.cat((x0_est_all, x0_est / mx), dim=0)
                x_dc_all = torch.cat((x_dc_all, x_ssdu / mx), dim=0)
                x0_error_all = torch.cat((x0_error_all, x_err / mx), dim=0)
                x_input_all = torch.cat((x_input_all, x_input / mx), dim=0)
            else:
                #x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
                x0_error_all = torch.cat((input_err / mx, x_err / mx), dim=0)
                # x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
                x_noisy = kspace_to_rss(y0 + noise1)
                # x0_est_all = torch.cat((x0 / mx, x_noisy / mx, x_input / mx, x0_est / mx), dim=0)
                # x0_est_all = torch.cat((x0 / mx, x_input / mx, x0_est / mx), dim=0)
                x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
                x_dc_all = torch.cat((x0 / mx, x_ssdu / mx), dim=0)
                x_input_all = x_input / mx


print('Noisy NMSE (k-space): {:e}'.format(nmse_loss(y0 + noise1, y0)))
print('Input SSIM: {:e}'.format(ssim(x1, np.array((m*x_input[0, 0]).detach().cpu()),
                     data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))
print('Noisy SSIM: {:e}'.format(ssim(x1, np.array((m*x_noisy[0, 0]).detach().cpu()),
                     data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))

x0_est_zoom = x0_est_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_dc_zoom = x_dc_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_input_zoom = x_input_all[:, :, xz:xz + x_wth, yz:yz + y_wth]

def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4, padding=1).detach().cpu()
    torchvision.utils.save_image(im, name)

print('seed is {}'.format(sd))
print('NMSEs are {} = {}dB'.format(all_nmse, 10*np.log10(all_nmse)))
print('SSIMs are {}'.format(all_ssim_hat))

x0_est_all = torchvision.utils.make_grid(x0_est_all, nrow=10).detach().cpu()
x0_est_zoom = torchvision.utils.make_grid(x0_est_zoom, nrow=10).detach().cpu()
x0_error_all = torchvision.utils.make_grid(x0_error_all, nrow=10).detach().cpu()

x_dc_all = torchvision.utils.make_grid(x_dc_all, nrow=10).detach().cpu()
x_dc_zoom = torchvision.utils.make_grid(x_dc_zoom, nrow=10).detach().cpu()

if use_slice_choice:
    save_loc = type + '/' + file_choice[:-3] + '_slice_' + str(slice_choice)
else:
    save_loc = type + '/' + str(sd)

print(save_loc)

if ~os.path.isdir(save_loc):
    os.makedirs(save_loc, exist_ok=True)

ndisp = 5
saveIm(x0_est_all, ndisp, save_loc + '/x.png')
saveIm(x_dc_all, ndisp, save_loc + '/x_dc.png')
saveIm(5*x0_error_all, ndisp,  save_loc + '/x_error.png')
saveIm(x0_est_zoom, ndisp, save_loc + '/x_zoom.png')
saveIm(x_dc_zoom, ndisp, save_loc + '/x_dc_zoom.png')
saveIm(below_noise_fl, ndisp, save_loc + '/below_noise_floor_.png')
saveIm(x_input_zoom, ndisp, save_loc + '/x_input_zoom.png')

# saveIm(torch.abs(y0_est[:,0,0, 220:420, 60:260]**2 + y0_est[:,1,0, 220:420, 60:260]**2 -
#                  y0[:,0,0, 220:420, 60:260]**2 - y0[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/kspace_error.png')

saveIm(torch.abs(y0_est[:,0,0]-y0[:,0,0])**0.1, ndisp, save_loc + '/kspace_error.png')


saveIm((y[:,0,0, 220:420, 60:260]**2 + y[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/y.png')
further_ss = y * mask_lambda
saveIm((further_ss[:,0,0, 220:420, 60:260]**2 + further_ss[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/y_tilde.png')
saveIm((y0_est[:,0,0, 220:420, 60:260]**2 + y0_est[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/y0_est.png')
saveIm(mask_lambda[:,0,0, 220:420, 60:260].float(), ndisp, save_loc + '/mask_lambda.png')
masked_noise = mask_omega * mask_lambda * noise2
saveIm((masked_noise[:,0,0, 220:420, 60:260]**2 + masked_noise[:,1,0, 220:420, 60:260]**2)*5 /sigma2, ndisp, save_loc + '/masked_noise.png')
noisy_y_tilde = further_ss + masked_noise
saveIm((noisy_y_tilde[:,0,0,220:420, 60:260]**2 + noisy_y_tilde[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/noisy_y_tilde.png')
