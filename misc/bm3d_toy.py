import torch.cuda
import torchvision

from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim

from data_loader.zf_data_loader import ZfData
from utils.preparations import *

import os
import itertools

from packages.bm3d import bm3d

sd = 320 # 260,  320, 420, 270. 570, 580
show_main_res = False

file_choice = 'file_brain_AXFLAIR_200_6002570.h5' # 'file_brain_AXFLAIR_200_6002548.h5'
#file_choice = '2022061207_FLAIR02.h5' #'2022062708_T102.h5'
slice_choice = 6
is_lowfield = False

use_slice_choice = True

accels = [4]
sigmas = [0.06]

type = 'bm3d/'  #+ str(accels[0]) + 'x_' + str(sigmas[0])

log_loc = [#'../logs/cuda/whole_experiment/4x/0.08/full.yaml/20649536',
            '../logs/cuda/whole_experiment/8x/0.06/ssdu.yaml/16408036']

xz = 120
yz = 60
x_wth = 100
y_wth = 100

all_ssim_hat = []
all_ssim_hat_tilde = []
all_nmse = []
all_nmse_tilde = []
correct_i = 'None'

with torch.no_grad():
    for l in log_loc:
        print('log directory: ' + l)

        if show_main_res:
            res = np.load(l + '/results.npz')
            print('Mean over test set is {}'.format(np.mean(res['loss'])))

        config = load_config(l + '/config')
        config['network']['device'] = 'cpu'
        # config['data']['us_fac_lambda'] = 1.2
        meth = config['data']['method']
        print('Method is: ' + meth)
        print('Accel is: ' + str(config['data']['us_fac']))

        pass_network, network = create_network(config)

        lambda_scale, alpha_param = prepare_hyp(network, config)
        optimizer = create_optimizer(network, config)
        criterion = create_criterion(config)

        network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
        network.to(config['network']['device'])
        network.eval()

        sigma1 = float(config['data']['sigma1'])
        if config['hyp_tuning']['alpha_active']:
            sigma2 = alpha_param[0] * sigma1
        else:
            sigma2 = torch.tensor(float(config['data']['sigma2']))

        if sigma1 > 0:
            alpha_sq = (sigma2 / sigma1) ** 2
        # alpha_sq = 1
        print('sigma1  is: ' + str(sigma1))
        print('sigma2  is: ' + str(sigma2))

        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=False)

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
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = mask_lambda*(y + mask_omega * noise2)
            outputs = pass_network(y_tilde, network)
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
        elif meth in ["n2n"]:
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = y + mask_omega * noise2
            outputs = pass_network(y_tilde, network)
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
        elif meth in ["ssdu", "ssdu_bern", "sure_ssdu"]:
            y0_est = pass_network(y, network)
            y_tilde = mask_lambda * y
            y0_est_tilde = pass_network(y_tilde, network)
        elif meth in ["full", "full_noisy", "noise2recon", "rei"]:
            outputs = pass_network(y, network)
            y0_est = outputs
            y0_est_tilde = outputs

        if meth not in ["full", "full_noisy", "noise2recon", "rei"]:
            print('Difference between y_tilde and y is {:e}'.format(torch.mean((y_tilde - y)**2)))
            print('Difference between number of non-zeros in y_tilde and y is {:e}'.format(torch.sum(mask_omega * mask_lambda) / torch.sum(mask_omega)))

        y0_est[pad] = 0  # set y0 padding to zero
        y0_est_tilde[pad] = 0
        x0_est = kspace_to_rss(y0_est)
        x0_est_tilde = kspace_to_rss(y0_est_tilde)
        x0 = kspace_to_rss(y0)

        # psd_range = np.linspace(0.01, 0.1, 10)
        # psd_nmse = []
        # for psd in psd_range:
        #     x0_est_denoi = bm3d(x0_est[0,0], psd)
        #     x0_est_denoi = torch.as_tensor(x0_est_denoi).unsqueeze(0).unsqueeze(0)
        #
        #     psd_nmse.append(nmse_loss(x0_est_denoi, x0).item())
        #     print(psd_nmse)

        x0_est_denoi = bm3d(x0_est[0, 0], 0.05)
        x0_est_denoi = torch.as_tensor(x0_est_denoi).unsqueeze(0).unsqueeze(0)

        print('Est NMSE denoi (image_domain): {:e}'.format(10*np.log10(nmse_loss(x0_est_denoi, x0))))

        pad_mask = x0 > 0.0

        x0_est = x0_est * pad_mask
        x0_est_tilde = x0_est_tilde * pad_mask
        x0 = x0 * pad_mask
        x0_est_denoi = x0_est_denoi * pad_mask

        print('Est NMSE (image domain): {:e}'.format(10*np.log10(nmse_loss(x0_est, x0))))
        print('Est NMSE: {:e}'.format(nmse_loss(y0_est, y0)))
        all_nmse.append(nmse_loss(y0_est, y0).item())
        all_nmse_tilde.append(nmse_loss(y0_est_tilde, y0).item())

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
        x1_tilde = np.array((m * x0_est_tilde[0, 0]).detach().cpu())
        all_ssim_hat.append(ssim(x1, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        all_ssim_hat_tilde.append(ssim(x1_tilde, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
        # all_ssim_hat.append(ssim(x1[xz:xz + x_wth, yz:yz + y_wth], x2[xz:xz + x_wth, yz:yz + y_wth],
        #                          data_range=x2.max()))

        if 'x0_est_all' in locals():
            x0_est_all = torch.cat((x0_est_all, x0_est / mx), dim=0)
            x0_est_all_denoi = torch.cat((x0_est_all_denoi, x0_est_denoi / mx), dim=0)
            x0_est_all_tilde = torch.cat((x0_est_all_tilde, x0_est_tilde / mx), dim=0)
            x_dc_all = torch.cat((x_dc_all, x_ssdu / mx), dim=0)
            x0_error_all = torch.cat((x0_error_all, x_err / mx), dim=0)
            x_input_all = torch.cat((x_input_all, x_input / mx), dim=0)
        else:
            #x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x0_error_all = torch.cat((input_err / mx, x_err / mx), dim=0)
            # x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x_noisy = kspace_to_rss(y0 + noise1)
            # x0_est_all = torch.cat((x0 / mx, x_noisy / mx, x_input / mx, x0_est / mx), dim=0)
            x0_est_all = torch.cat((x0 / mx, x_input / mx, x0_est / mx), dim=0)
            x0_est_all_denoi = torch.cat((x0 / mx, x_input / mx, x0_est_denoi / mx), dim=0)
            # x0_est_all = torch.cat((x0 / mx, x0_est / mx), dim=0)
            x_dc_all = torch.cat((x0 / mx, x_ssdu / mx), dim=0)
            x0_est_all_tilde = torch.cat((x0 / mx, x0_est_tilde / mx), dim=0)
            x_input_all = x_input / mx


print('Noisy NMSE (k-space): {:e}'.format(nmse_loss(y0 + noise1, y0)))
print('Input SSIM: {:e}'.format(ssim(x1, np.array((m*x_input[0, 0]).detach().cpu()),
                     data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))
print('Noisy SSIM: {:e}'.format(ssim(x1, np.array((m*x_noisy[0, 0]).detach().cpu()),
                     data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)))

x0_est_zoom = x0_est_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x0_est_zoom_tilde = x0_est_all_tilde[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_dc_zoom = x_dc_all[:, :, xz:xz + x_wth, yz:yz + y_wth]
x_input_zoom = x_input_all[:, :, xz:xz + x_wth, yz:yz + y_wth]


def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4, padding=1).detach().cpu()
    torchvision.utils.save_image(im, name)

print('seed is {}'.format(sd))
print('NMSEs are {} = {}dB'.format(all_nmse, 10*np.log10(all_nmse)))
print('NMSEs for tildes are {} = {}dB'.format(all_nmse_tilde, 10*np.log10(all_nmse_tilde)))
print('SSIMs are {}'.format(all_ssim_hat))
print('SSIMs for tildes are {}'.format(all_ssim_hat_tilde))

x0_est_all = torchvision.utils.make_grid(x0_est_all, nrow=10).detach().cpu()
x0_est_all_denoi = torchvision.utils.make_grid(x0_est_all_denoi, nrow=10).detach().cpu()
x0_est_all_tilde = torchvision.utils.make_grid(x0_est_all_tilde, nrow=10).detach().cpu()
x0_est_zoom = torchvision.utils.make_grid(x0_est_zoom, nrow=10).detach().cpu()
x0_est_zoom_tilde = torchvision.utils.make_grid(x0_est_zoom_tilde, nrow=10).detach().cpu()
x0_error_all = torchvision.utils.make_grid(x0_error_all, nrow=10).detach().cpu()

x_dc_all = torchvision.utils.make_grid(x_dc_all, nrow=10).detach().cpu()
x_dc_zoom = torchvision.utils.make_grid(x_dc_zoom, nrow=10).detach().cpu()

if use_slice_choice:
    save_loc = '../saved_images/' + type + '/' + file_choice[:-3] + '_slice_' + str(slice_choice)
else:
    save_loc = '../saved_images/' + type + '/' + str(sd)

print(save_loc)

if ~os.path.isdir(save_loc):
    os.makedirs(save_loc, exist_ok=True)

ndisp = 5
saveIm(x0_est_all, ndisp, save_loc + '/x.png')
saveIm(x0_est_all_denoi, ndisp, save_loc + '/x_denoi.png')
saveIm(x0_est_all_tilde, ndisp, save_loc + '/x_tilde.png')
saveIm(x_dc_all, ndisp, save_loc + '/x_dc.png')
saveIm(5*x0_error_all, ndisp,  save_loc + '/x_error.png')
saveIm(x0_est_zoom, ndisp, save_loc + '/x_zoom.png')
saveIm(x0_est_zoom_tilde, ndisp, save_loc + '/x_zoom_tilde.png')
saveIm(x_dc_zoom, ndisp, save_loc + '/x_dc_zoom.png')
# saveIm(below_noise_fl, ndisp, save_loc + '/below_noise_floor_.png')
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
