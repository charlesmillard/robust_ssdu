from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from utils_n2n.preparations import *

import os

print(os.getcwd())

type = "n2n_ssdu"
if type == 'test':
    root = 'logs/cuda/4x/'
    # log_loc = [root + '75818101', root + '76024129' ]
    sig = '0.04'
    log_loc = [root + 'n2n_ssdu/76846908_' + sig]
elif type == "ssdu":
    root = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/4x/n2n_ssdu/'
    log_loc = [root + '76846908_0.02/',root + '76846908_0.04/',root + '76846908_0.06/',root + '76846908_0.08/']
elif type == "n2n_ssdu":
    root = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/denoising_proxy_test/'
    log_loc = [root + "n2n_ssdu1.5.yaml/81196193", root + "n2n_ssdu2.yaml/81196193",
               root + "n2n_ssdu2.5.yaml/81198029", root + "n2n_ssdu4.yaml/81196193",
               root + "n2n_ssdu6.yaml/81196193"]


xz = 120
yz = 60
wth = 100

loss_track = []
actual_loss = []
loss_nonorm = []
lambda_list = []
im_actual_loss = []

lmb_idx = 0
with torch.no_grad():
    for l in log_loc:
        print('log directory: ' + l)
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

        sd = 440 # 260,  320, 420
        np.random.seed(sd)
        torch.manual_seed(sd)
        torch.cuda.manual_seed_all(sd)

        lambda_list.append(config['data']['us_fac_lambda'])
        # for lmb_idx in range(len(lambda_list)):
        #     lmb = lambda_list[lmb_idx]
        loss_track.append([])
        actual_loss.append([])
        loss_nonorm.append([])
        im_actual_loss.append([])
        # config['data']['us_fac_lambda'] = lmb
        test_load = DataLoader(ZfData('test', config), batch_size=20, shuffle=True)
        network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
        network.to(config['network']['device'])
        network.eval()
        for i, data in enumerate(test_load, 0):
            y0, noise1, noise2, mask_omega, mask_lambda, mu, mask = data
            # print(torch.mean(mask_omega.float()))
            mu = mu[0] if meth == "n2n_ssdu" else 1  # all the same

            noise1 *= sigma1
            noise2 *= sigma2
            y = mask_omega * (y0 + noise1)

            net = lambda y: pass_varnet(y, network)

            below_noise_fl = torch.as_tensor(torch.abs(y0[0, 0, 0]) < sigma1, dtype=float)
            print(torch.mean(below_noise_fl))
            print(meth, sigma1)

            pad = torch.abs(y0) == 0

            if meth in ["n2n_ssdu"]:
                #y_tilde = y # * mask_lambda
                #y_tilde = mask_lambda*y
                y_tilde = mask_lambda*(y + mask_omega * noise2)
                # y_tilde = y
                outputs = net(y_tilde)
                y0_est = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)

            elif meth in ["n2n"]:
                y_tilde = y
                outputs = net(y_tilde)
                y0_est = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
            elif meth in ["ssdu", "ssdu_bern", "sure_ssdu"]:
                y_tilde = mask_lambda * y  # second mask
                # outputs = pass_network(y_tilde, network)
                # y0_est = outputs * (y == 0) + y
                y0_est = net(y)
            elif meth in ["full", "full_noisy", "noise2recon", "rei"]:
                y_tilde = y
                outputs = net(y)
                y0_est = outputs

            y0_est[pad] = 0

            denoi_mask = (y_tilde != 0)
            sure = sure_loss(net, y_tilde, outputs, denoi_mask.float(), 1e-8, torch.as_tensor(sigma1))
            loss_track[lmb_idx].append(sure * torch.mean(torch.abs(y_tilde * denoi_mask)**2))
            loss_nonorm[lmb_idx].append(
                mse_loss(outputs * denoi_mask, y0 * denoi_mask) * torch.mean(torch.abs(y0 * denoi_mask) ** 2))
            # actual_loss[lmb_idx].append(
            #     mse_loss(outputs, y0) * torch.mean(torch.abs(y0) ** 2))

            actual_loss[lmb_idx].append(torch.mean(torch.abs(outputs - y0)**2))
            # loss_track[lmb_idx].append(torch.sum((1-mask_lambda) * mask_omega * (y0_est - y0) ** 2)/norm)
            # loss_nonorm[lmb_idx].append(torch.sum((1 - mask_lambda) * mask_omega * (y0_est - y0) ** 2) )

            x0_est = kspace_to_rss(y0_est)
            x0 = kspace_to_rss(y0)

            im_actual_loss[lmb_idx].append(mse_loss(x0_est, x0) * torch.mean(torch.abs(x0) ** 2))

            if i == 0:
                break
        lmb_idx += 1

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(141)
plt.plot(lambda_list, np.mean(actual_loss, axis=1), '-x')
plt.title('Actual k-space error')
plt.xlabel('R tilde')
plt.subplot(142)
plt.plot(lambda_list, np.mean(loss_nonorm, axis=1),'-x')
plt.title('Error on sampled coeffs')
plt.xlabel('R tilde')
plt.subplot(143)
plt.plot(lambda_list, np.mean(loss_track , axis=1) , '-x')
plt.xlabel('R tilde')
plt.title('SURE on sampled coeffs')
plt.subplot(144)
plt.plot(lambda_list, np.mean(im_actual_loss , axis=1) , '-x')
plt.xlabel('R tilde')
plt.title('Error on cropped magnitude image')
plt.show()

# plt.figure()
# plt.plot(lambda_list, np.mean(actual_loss, axis=1), '-x')
# plt.plot(lambda_list, np.mean(loss_track, axis=1), '-x')
# plt.legend(['actual loss', 'SURE estimate'])
# plt.xlabel('R tilde')
# plt.show()
