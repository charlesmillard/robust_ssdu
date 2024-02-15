import torch.cuda
import torchvision

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from data_loader.zf_data_loader import ZfData
from utils_n2n.preparations import *
from model.mask_tools import *
from configs.config_loader import *

from packages.bm3d import bm3d

import os
import itertools

sd = 320  # 260,  320, 420, 270. 570, 580
show_main_res = False

file_options = ['file_brain_AXFLAIR_200_6002566.h5',
                'file_brain_AXFLAIR_200_6002585.h5',
                'file_brain_AXFLAIR_200_6002570.h5',
                'file_brain_AXFLAIR_200_6002558.h5', #craniotomy example
                'file_brain_AXFLAIR_200_6002567.h5',
                'file_brain_AXFLAIR_200_6002549.h5']

file_choice = file_options[0]
# file_choice = '2022121708_T104.h5' # '2022121708_FLAIR04.h5' #'2022121708_T203.h5' # '2022120404_T201.h5' #'2022120409_FLAIR02.h5' # '2022101303_T104.h5'
slice_choice = 6
is_lowfield = False

use_slice_choice = True
verbose = False

accels = [4]
sigmas = [0.08]
type = 'new_weighting/pathologies/' + str(accels[0]) + 'x_' + str(sigmas[0])

type = 'new_weighting/r2r/'

roots = [  # 'just_n2n_ssdu_alph0.5',
    # 'whole_experiment/8x/0.06/noise2recon.yaml',
    # 'whole_experiment_alph1',
    # 'whole_experiment_alph1/4x/0.02/noise2recon.yaml',
    # 'whole_experiment_alph1/4x/0.04/noise2recon.yaml',
    'whole_experiment_alph1/4x/0.06/noise2recon.yaml',
    'whole_experiment_alph1/4x/0.08/noise2recon.yaml',
    # 'whole_experiment_alph1/8x/0.02/noise2recon.yaml',
    # 'whole_experiment_alph1/8x/0.04/noise2recon.yaml',
    # 'whole_experiment_alph1/8x/0.06/noise2recon.yaml',
    'whole_experiment_alph1/8x/0.08/noise2recon.yaml',
    # 'alpha_robustness_weighted_only/1/',
    # 'old_whole_experiment_alph1_weighted',
    'alpha_robustness_n2n_ssdu_weighted/0.5.yaml'
    # 'new_format',
    'whole_experiment_alph1_weighted',
    'whole_experiment_alph1_weighted_n2n_ssdu',
    'whole_experiment_alph1/8x/0.08/noise2recon.yaml',
    'whole_experiment_alph1_weighted',
    'whole_experiment']

roots = [# 'just_n2n_ssdu_alph0.5',
         # 'whole_experiment/8x/0.06/noise2recon.yaml',
         # 'whole_experiment_alph1',
        #'whole_experiment_alph1/4x/0.02/noise2recon.yaml',
        # 'whole_experiment_alph1/4x/0.04/noise2recon.yaml',
        'alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38770026_5',
        'alpha_robustness_weighted_only/1/n2n_weighted.yaml/30265058',
        'new_format/rssdu_alph75',
        'new_format/unw_n2f_alph125',
        'new_format/unw_rssdu_alph50',
         'whole_experiment_alph1/4x/0.06/noise2recon.yaml',
        'whole_experiment_alph1/4x/0.08/noise2recon.yaml',
        # 'whole_experiment_alph1/8x/0.02/noise2recon.yaml',
        'whole_experiment_alph1/8x/0.04/noise2recon.yaml',
        #'whole_experiment_alph1/8x/0.06/noise2recon.yaml',
        'alpha_robustness_weighted_only',
        'whole_experiment_alph1/8x/0.08/noise2recon.yaml',
        'whole_experiment_alph1_weighted',
        'whole_experiment']

root_loc = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/'
# meths = ['full', 'full_noisy', 'n2n', 'n2n', 'ssdu', 'noise2recon', 'rei',  'n2n_ssdu',  'n2n_ssdu']
# is_weighted = [None, None, False, True, None, None, None, False,  True]

meths = ['full', 'noise2full', 'noisier2full', 'noisier2full', 'ssdu', 'noise2recon', 'robust_ssdu', 'robust_ssdu']
is_weighted = [None, None, False, True, None, None, False, True]

# subset = (0, 1, 2, 3, 4, 5, 6, 7) # everything
#subset = (0, 1, 2, 3) # noisy, fully sampled training data
subset = (0, 4, 5, 6, 7)  # noisy, sub-sampled sampled training data
subset = (4, 7)

meths = [meths[i] for i in subset]
is_weighted = [is_weighted[i] for i in subset]

n_accels = len(accels)
n_sigmas = len(sigmas)
n_meths = len(meths)

all_logs = []
for r in roots:
    for f in os.walk(root_loc + r):
        if 'state_dict' in f[2]:
            all_logs.append(f[0])

log_loc = np.zeros((n_accels, n_sigmas, n_meths), dtype=object)

for log in all_logs:
    config = load_config(log + '/config')
    config = reformat_config(config)

    ac = config['mask']['us_fac']
    sig = config['noise']['sigma1']
    meth = config['optimizer']['method']
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

log_loc = np.reshape(log_loc, (n_accels * n_sigmas * n_meths))

bm3d_denoising = [False, False, False]  # [False, False, True, False]

# log_loc = ['saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#            'saved/logs/cuda/alpha_robustness/1/n2n.yaml/20391645',
#            'saved/logs/cuda/alpha_robustness_weighted_only/1/n2n_weighted.yaml/30265058']
#            #'saved/logs/cuda/whole_experiment_alph1_weighted/8x/0.06/n2n.yaml/34363466']
#
# log_loc = ['saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#            'saved/logs/cuda/whole_experiment/8x/0.06/ssdu.yaml/16408036',
#            'saved/logs/cuda/whole_experiment_alph1/8x/0.06/n2n_ssdu.yaml/22951552',
#            'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/8x/0.06/n2n_ssdu.yaml/36152790_1',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38133993_1_rootalpha']
#
# log_loc = ['saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#            'saved/logs/cuda/new_format/full.yaml/38200612_1']
#
# log_loc = ['saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#             'saved/logs/cuda/alpha_robustness_n2n_ssdu/0.25/n2n_ssdu_weighted.yaml/24202007',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.25.yaml/38761703_3']

# log_loc = ['saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.75.yaml/37921353_5',
#             'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38770026_5']
# log_loc = ['saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.25.yaml/37816028_3',
#             'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.5.yaml/37464314_2',
#            'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/8x/0.06/n2n_ssdu.yaml/36152790_1',
#             'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/1.25.yaml/37921352_4',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/1.75.yaml/37152032_6']

# log_loc = ['saved/logs/cuda/whole_experiment_alph1_weighted/4x/0.04/n2n.yaml/34554803']

# log_loc = ['saved/logs/cuda/whole_experiment/8x/0.04/full.yaml/17116723',
#            'saved/logs/cuda/whole_experiment_alph1/8x/0.04/n2n_ssdu.yaml/23090715',
#            'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/8x/0.04/n2n_ssdu.yaml/36565386_4']

# log_loc = ['logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984',
#            'logs/cuda/alpha_robustness_weighted_only/0.25/n2n_weighted.yaml/29865869',
#             'logs/cuda/alpha_robustness_weighted_only/1/n2n_weighted.yaml/30265058']

# log_loc = ['saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.5.yaml/38761182_2',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38770026_5',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/1.5.yaml/38761181_1']
#
# log_loc = ['saved/logs/cuda/new_format/ssdu_high_lr.yaml/39380227_1']
#
# log_loc = [ 'saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984/',
#             'saved/logs/cuda/whole_experiment/8x/0.06/n2n_ssdu.yaml/16421958',
#            'saved/logs/cuda/alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38770026_5',
#            'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/8x/0.06/n2n_ssdu.yaml/39466704_1']

log_loc = ['saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984/',
           'saved/logs/cuda/whole_experiment/8x/0.06/n2n_ssdu.yaml/16421958',
           'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/8x/0.06/n2n_ssdu.yaml/39466704_1',
           'saved/logs/cuda/new_format/r2r_ssdu_rob/r2r_ssdu_075.yaml/46239147_6',
           ]

# log_loc = ['saved/logs/cuda/whole_experiment/4x/0.04/full.yaml/16815717',
#            'saved/logs/cuda/whole_experiment_alph1_weighted_n2n_ssdu/4x/0.04/n2n_ssdu.yaml/38969338_8',
#            'saved/logs/cuda/new_format/r2r_ssdu_4x.yaml/45713543_1',
#            'saved/logs/cuda/new_format/r2r_ssdu_4x_kw.yaml/45744120_1']



print(log_loc)

n_diff = len(log_loc) - len(bm3d_denoising)
if n_diff > 0:
    for jj in range(n_diff):
        bm3d_denoising.append(False)

xz = 120
yz = 60
x_wth = 100
y_wth = 100

# xz = 80
# yz = 40
# x_wth = 100
# y_wth = 100

# xz = 148
# yz = 178
# x_wth = 17
# y_wth = 25

# xz = 83
# yz = 106
# x_wth = 23
# y_wth = 40

all_ssim_hat = []
all_ssim_hat_tilde = []
all_nmse = []
all_nmse_rss = []
all_nmse_tilde = []
correct_i = 'None'

bm3d_idx = 0
slice_found = False

with torch.no_grad():
    for l in log_loc:
        print('********')
        print('log directory: ' + l)

        if show_main_res:
            res = np.load(l + '/results.npz')
            print('Mean over test set is {}'.format(np.mean(res['loss'])))

        config = load_config(l + '/config')
        config = reformat_config(config)
        config = set_missing_config_entries(config)
        if config['data']['loc'] == 'auto':
            config['data']['loc'] = data_root(config)

        config['network']['device'] = 'cpu'
        config['data']['loc'] = '/home/xsd618/data/fastMRI_test_subset_brain/'

        config['data']['fixed_ncoil'] = None

        meth = config['optimizer']['method']
        print('Method is: ' + meth)
        print('Accel is: ' + str(config['mask']['us_fac']))

        pass_network, network = create_network(config)

        alpha = torch.tensor(config['noise']['alpha'])
        alpha_sq = alpha ** 2
        lambda_scale = config['mask']['us_fac_lambda']

        optimizer = create_optimizer(network, config)
        criterion = create_criterion(config)

        network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
        network.to(config['network']['device'])
        network.eval()

        sigma1 = float(config['noise']['sigma1'])
        sigma2 = alpha * sigma1

        print('sigma1  is: ' + str(sigma1))
        print('sigma2  is: ' + str(sigma2))

        sd = 420  # 260,  320, 420
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        # set_seeds(config)

        test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=False)

        if use_slice_choice:
            if correct_i == 'None':
                for i, data in enumerate(test_load, 0):
                    slice_info = data[-1]
                    print(slice_info)
                    is_correct_slice = (slice_info[0][0] == file_choice and int(slice_info[1]) == slice_choice)
                    if is_correct_slice:
                        correct_i = i
                        slice_found = True
                        break
        else:
            correct_i = np.random.randint(254)
            slice_found = True

        if not slice_found:
            raise Exception('Slice not found')

        y0, noise1, mask_omega, prob_omega, mask, slice_info = \
            next(itertools.islice(test_load, correct_i, None))


        def zero_k(k):
            return zero_kspace(k, torch.abs(y0) == 0)


        mask_lambda, kweight = make_lambda_mask(prob_omega, lambda_scale, config)
        noise2 = torch.randn(y0.shape)

        noise1 = zero_k(noise1)
        noise2 = zero_k(noise2)

        mask_omega = zero_k(mask_omega)
        mask_lambda = zero_k(mask_lambda)

        noise1 *= sigma1
        noise2 *= sigma2

        if config['noise']['sim_noise']:
            y = mask_omega * (y0 + noise1)
        else:
            y = mask_omega * y0

        if config['data']['set'] == 'm4raw':
            x0_unaveraged = kspace_to_rss(y0)
            x0 = m4raw_averages(file_choice, slice_choice, config)
            x0 = x0[0]
            pad_mask = x0 > torch.max(x0) / 6
        else:
            x0 = kspace_to_rss(y0)
            x0_unaveraged = x0
            pad_mask = 1

        below_noise_fl = torch.as_tensor(torch.abs(y0[0, 0, 0]) < sigma1, dtype=float)

        # pad = torch.abs(y0) == 0
        y = zero_k(y)

        if meth == "robust_ssdu":
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = mask_lambda * (y + mask_omega * noise2)
            outputs = pass_network(y_tilde, network)
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
        elif meth == "noisier2full":
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = y + mask_omega * noise2
            outputs = pass_network(y_tilde, network)
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
        elif meth in ["ssdu", "ssdu_bern", "sure_ssdu"]:
            y0_est = pass_network(y, network)
            y0_est = y0_est * (y == 0) + y * (y != 0)

            y_tilde = mask_lambda * y
            y0_est_tilde = pass_network(y_tilde, network)
            y0_est_tilde = y0_est_tilde * (y == 0) + y * (y != 0)
        elif meth in ["full", "noise2full", "noise2recon", "rei"]:
            outputs = pass_network(y, network)
            y0_est = outputs
            y0_est_tilde = outputs
        elif meth == "r2r_ssdu":
            outputs = pass_network(y, network)
            y0_est = (y != 0) * outputs + outputs * (y == 0)

            y_tilde = mask_lambda * (y + mask_omega * noise2)
            outputs = pass_network(y_tilde, network)
            y0_est_tilde = (y_tilde != 0) * outputs + outputs * (y_tilde == 0)



        # if meth not in ["full", "noise2full", "noise2recon", "rei"]:
        #     print('Difference between y_tilde and y is {:e}'.format(torch.mean((y_tilde - y)**2)))
        #     print('Difference between number of non-zeros in y_tilde and y is {:e}'.format(torch.sum(mask_omega * mask_lambda) / torch.sum(mask_omega)))

        # y0 = unwhiten_kspace(file_choice, slice_choice, y0[0], config).unsqueeze(0)
        # y0_est = unwhiten_kspace(file_choice, slice_choice, y0_est[0], config).unsqueeze(0)
        # y0_est_tilde = unwhiten_kspace(file_choice, slice_choice, y0_est_tilde[0], config).unsqueeze(0)

        y0_est = zero_k(y0_est)
        y0_est_tilde = zero_k(y0_est_tilde)

        x0_est = kspace_to_rss(y0_est)
        x0_est_tilde = kspace_to_rss(y0_est_tilde)
        x_input = kspace_to_rss(y)

        if bm3d_denoising[bm3d_idx]:
            print('Including BM3D')
            x0_est = bm3d(x0_est[0, 0], sigma_psd=torch.std(x0_est - x0))
            x0_est = torch.as_tensor(x0_est).unsqueeze(0).unsqueeze(0)

        bm3d_idx += 1

        x0_est = x0_est * pad_mask
        x0_est_tilde = x0_est_tilde * pad_mask
        x0 = x0 * pad_mask
        x0_unaveraged = x0_unaveraged * pad_mask
        x_input = x_input * pad_mask

        if config['data']['set'] == 'm4raw':
            x0 = torch.flip(x0, [2])
            x0_est_tilde = torch.flip(x0_est_tilde, [2])
            x0_est = torch.flip(x0_est, [2])
            x0_unaveraged = torch.flip(x0_unaveraged, [2])
            x_input = torch.flip(x_input, [2])

        if verbose:
            print('Est NMSE: {:e}'.format(nmse_loss(y0_est, y0)))
            print('Est NMSE (im): {:e}'.format(nmse_loss(x0_est, x0)))

        all_nmse.append(nmse_loss(y0_est, y0).item())
        all_nmse_rss.append(nmse_loss(x0_est, x0).item())
        all_nmse_tilde.append(nmse_loss(y0_est_tilde, y0).item())

        mx = torch.max(torch.abs(x0[0]))
        # print(mx)

        y_ssdu = (y != 0) * y + y0_est * (y == 0)
        y_ssdu = zero_k(y_ssdu)
        # y_ssdu[pad] = 0
        x_ssdu = kspace_to_rss(y_ssdu)

        # x_err = kspace_to_rss(y0_est - y0)
        # input_err = kspace_to_rss(y - y0)

        x_err = torch.abs(x0_est - x0)
        input_err = torch.abs(x_input - x0)

        nx = y.shape[-2]
        if torch.numel(mask < 10):
            mask = kspace_to_im(y0) > 0  # 0.2

        crop_sz = min([160, config['data']['nx'] // 2])
        m = mask[0, 0, 0, nx // 2 - crop_sz:nx // 2 + crop_sz]

        x_im = kspace_to_im(y0_est)
        masked_im = mask * x_im
        masked_im = masked_im[masked_im != 0]

        print('Input NMSE (k-space): {:e}'.format(nmse_loss(y, y0)))
        if verbose:
            print('Input NMSE (image_domain): {:e}'.format(nmse_loss(x_input, x0)))
            print('Unaveraged NMSE (image_domain): {:e}'.format(nmse_loss(x0_unaveraged, x0)))
            print('Estimate of noise std is {}'.format(torch.std(masked_im)))

        x2 = np.array((m * x0[0, 0]).detach().cpu())
        x1 = np.array((m * x0_est[0, 0]).detach().cpu())
        x1_tilde = np.array((m * x0_est_tilde[0, 0]).detach().cpu())
        all_ssim_hat.append(
            ssim(x1, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        all_ssim_hat_tilde.append(
            ssim(x1_tilde, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        # all_ssim_hat.append(ssim(x1[xz:xz + x_wth, yz:yz + y_wth], x2[xz:xz + x_wth, yz:yz + y_wth],
        #                          data_range=x2.max()))

        if 'x0_est_all' in locals():
            x0_est_all = torch.cat((x0_est_all, x0_est / mx), dim=0)
            x0_est_all_tilde = torch.cat((x0_est_all_tilde, x0_est_tilde / mx), dim=0)
            x_dc_all = torch.cat((x_dc_all, x_ssdu / mx), dim=0)
            x0_error_all = torch.cat((x0_error_all, x_err / mx), dim=0)
            x_input_all = torch.cat((x_input_all, x_input / mx), dim=0)
        else:
            # x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x0_error_all = torch.cat((input_err / mx, x_err / mx), dim=0)
            # x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x_noisy = kspace_to_rss(y0 + noise1)
            #x0_est_all = torch.cat((x0 / mx, x_noisy / mx, x_input / mx, x0_est / mx), dim=0)
            # x0_est_all = torch.cat((x0 / mx, x_input / mx, x0_est / mx), dim=0)
            x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x_dc_all = torch.cat((x0 / mx, x_ssdu / mx), dim=0)
            x0_est_all_tilde = torch.cat((x0 / mx, x_input / mx, x0_est_tilde / mx), dim=0)
            x_input_all = x_input / mx

print('Noisy NMSE (k-space): {:e}'.format(nmse_loss(y0 + noise1, y0)))
print('Input SSIM: {:e}'.format(ssim(x1, np.array((m * x_input[0, 0]).detach().cpu()),
                                     data_range=x2.max(), gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False)))
print('Noisy SSIM: {:e}'.format(ssim(x1, np.array((m * x_noisy[0, 0]).detach().cpu()),
                                     data_range=x2.max(), gaussian_weights=True, sigma=1.5,
                                     use_sample_covariance=False)))

x0_est_zoom = x0_est_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x0_est_zoom_tilde = x0_est_all_tilde[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_dc_zoom = x_dc_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_input_zoom = x_input_all[:, :, xz:xz + x_wth, yz:yz + y_wth]


def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4, padding=1).detach().cpu()
    torchvision.utils.save_image(im, name)


print('seed is {}'.format(sd))
print('NMSEs are {} = {}dB'.format(all_nmse, 10 * np.log10(all_nmse)))
print('NMSEs for tildes are {} = {}dB'.format(all_nmse_tilde, 10 * np.log10(all_nmse_tilde)))
print('NMSEs for RSS are {} = {}dB'.format(all_nmse_rss, 10 * np.log10(all_nmse_rss)))
print('SSIMs are {}'.format(all_ssim_hat))
print('SSIMs for tildes are {}'.format(all_ssim_hat_tilde))

x0_est_all = torchvision.utils.make_grid(x0_est_all, nrow=10).detach().cpu()
x0_est_all_tilde = torchvision.utils.make_grid(x0_est_all_tilde, nrow=10).detach().cpu()
x0_est_zoom = torchvision.utils.make_grid(x0_est_zoom, nrow=10).detach().cpu()
x0_est_zoom_tilde = torchvision.utils.make_grid(x0_est_zoom_tilde, nrow=10).detach().cpu()
x0_error_all = torchvision.utils.make_grid(x0_error_all, nrow=10).detach().cpu()

x_dc_all = torchvision.utils.make_grid(x_dc_all, nrow=10).detach().cpu()
x_dc_zoom = torchvision.utils.make_grid(x_dc_zoom, nrow=10).detach().cpu()

if use_slice_choice:
    save_loc = 'saved/saved_images/' + type + '/' + file_choice[:-3] + '_slice_' + str(slice_choice)
else:
    save_loc = 'saved/saved_images/' + type + '/' + str(sd)

print(save_loc)

if ~os.path.isdir(save_loc):
    os.makedirs(save_loc, exist_ok=True)

ndisp = 5
saveIm(x0_est_all, ndisp, save_loc + '/x.png')
saveIm(x0_est_all_tilde, ndisp, save_loc + '/x_tilde.png')
saveIm(x_dc_all, ndisp, save_loc + '/x_dc.png')
saveIm(5 * x0_error_all, ndisp, save_loc + '/x_error.png')
saveIm(x0_est_zoom, ndisp, save_loc + '/x_zoom.png')
saveIm(x0_est_zoom_tilde, ndisp, save_loc + '/x_zoom_tilde.png')
saveIm(x_dc_zoom, ndisp, save_loc + '/x_dc_zoom.png')
# saveIm(below_noise_fl, ndisp, save_loc + '/below_noise_floor_.png')
saveIm(x_input_zoom, ndisp, save_loc + '/x_input_zoom.png')

# saveIm(torch.abs(y0_est[:,0,0, 220:420, 60:260]**2 + y0_est[:,1,0, 220:420, 60:260]**2 -
#                  y0[:,0,0, 220:420, 60:260]**2 - y0[:,1,0, 220:420, 60:260]**2)**0.1, ndisp, save_loc + '/kspace_error.png')

saveIm(torch.abs(y0_est[:, 0, 0] - y0[:, 0, 0]) ** 0.1, ndisp, save_loc + '/kspace_error.png')

saveIm((y[:, 0, 0, 220:420, 60:260] ** 2 + y[:, 1, 0, 220:420, 60:260] ** 2) ** 0.1, ndisp, save_loc + '/y.png')
further_ss = y * mask_lambda
saveIm((further_ss[:, 0, 0, 220:420, 60:260] ** 2 + further_ss[:, 1, 0, 220:420, 60:260] ** 2) ** 0.1, ndisp,
       save_loc + '/y_tilde.png')
saveIm((y0[:, 0, 0, 220:420, 60:260] ** 2 + y0[:, 1, 0, 220:420, 60:260] ** 2) ** 0.1, ndisp, save_loc + '/y0.png')
saveIm((y0_est[:, 0, 0, 220:420, 60:260] ** 2 + y0_est[:, 1, 0, 220:420, 60:260] ** 2) ** 0.1, ndisp,
       save_loc + '/y0_est.png')
saveIm(mask_lambda[:, 0, 0, 220:420, 60:260].float(), ndisp, save_loc + '/mask_lambda.png')
masked_noise = noise2  # mask_omega * mask_lambda * noise2
saveIm((masked_noise[:, 0, 0, 220:420, 60:260] ** 2 + masked_noise[:, 1, 0, 220:420, 60:260] ** 2) * 5 / sigma2, ndisp,
       save_loc + '/masked_noise.png')
noisy_y_tilde = further_ss + masked_noise
saveIm((noisy_y_tilde[:, 0, 0, 220:420, 60:260] ** 2 + noisy_y_tilde[:, 1, 0, 220:420, 60:260] ** 2) ** 0.1, ndisp,
       save_loc + '/noisy_y_tilde.png')
