from torch.utils.data import DataLoader

from data_loader.zf_data_loader import ZfData
from utils.preparations import *

from utils.mask_tools import *

import itertools

sd = 320 # 260,  320, 420, 270. 570, 580
show_main_res = False

file_choice = 'file_brain_AXFLAIR_200_6002549.h5'
#file_choice = '2022061207_FLAIR02.h5' #'2022062708_T102.h5'
slice_choice = 6
is_lowfield = False

use_slice_choice = True

log_loc = ['../logs/cuda/alpha_robustness/0.25/n2n_weighted.yaml/20391283',
           '../logs/cuda/weighted_experiment/8x/0.06/n2n.yaml/17614579',
           '../logs/cuda/alpha_robustness/0.75/n2n_weighted.yaml/20396269',
           '../logs/cuda/alpha_robustness/1/n2n_weighted.yaml/20391458',
           '../logs/cuda/alpha_robustness/1.25/n2n_weighted.yaml/22518839',
           '../logs/cuda/alpha_robustness/1.5/n2n_weighted.yaml/22591989',
           '../logs/cuda/alpha_robustness/1.75/n2n_weighted.yaml/22549561',
           '../logs/cuda/alpha_robustness/2/n2n_weighted.yaml/22549541']

log_loc = ['../logs/cuda/alpha_robustness_n2n_ssdu/0.25/n2n_ssdu_weighted.yaml/24202007',
           '../logs/cuda/weighted_experiment/8x/0.06/n2n_ssdu.yaml/17584179',
           '../logs/cuda/alpha_robustness_n2n_ssdu/0.75/n2n_ssdu_weighted.yaml/23697309',
           '../logs/cuda/alpha_robustness_n2n_ssdu/1.25/n2n_ssdu_weighted.yaml/24202005',
           '../logs/cuda/alpha_robustness_n2n_ssdu/1.5/n2n_ssdu_weighted.yaml/24271648',
           '../logs/cuda/alpha_robustness_n2n_ssdu/1.75/n2n_ssdu_weighted.yaml/24260111',
           '../logs/cuda/alpha_robustness_n2n_ssdu/2/n2n_ssdu_weighted.yaml/24206192']

denoi_loss_est_all = []
denoi_loss_est_actual = []
denoi_loss_no_correction = []
denoi_loss_old = []
alpha_all = []

recon_loss_est_all = []
recon_loss_est_actual = []
recon_loss_no_correction = []

correct_i = 'None'

with (torch.no_grad()):
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
            alpha_weight = ((1 + alpha_sq) / alpha_sq)

        alpha_all.append(alpha_sq**0.5)

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

        y0, noise1, noise2, mask_omega, mask_lambda, Kweight, prob_omega, mask, slice_info = \
            next(itertools.islice(test_load, correct_i, None))

        noise1 *= sigma1
        noise2 *= sigma2

        y = mask_omega * (y0 + noise1)

        pad = torch.abs(y0) == 0

        pad_reshaped = pad[:,0:1,0:1]
        mask_omega[pad_reshaped] = 0
        mask_lambda[pad_reshaped] = 0
        noise1[pad] = 0
        noise2[pad] = 0
        y[pad] = 0

        if meth in ["n2n_ssdu"]:
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = mask_lambda*(y + mask_omega * noise2)
            outputs = pass_network(y_tilde, network)
            outputs[pad] = 0
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
        elif meth in ["n2n"]:
            outputs = pass_network(y, network)
            y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)

            y_tilde = y + mask_omega * noise2
            outputs = pass_network(y_tilde, network)
            outputs[pad] = 0
            y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)

        y0_est_tilde[pad] = 0
        outputs[pad] = 0

        total_loss_actual = mse_loss(y0_est_tilde, y0)

        if meth == 'n2n':
            recon_mask = (1 - mask_omega)
            recon_mask[pad_reshaped] = 0
            recon_loss_actual = mse_loss(recon_mask * y0_est_tilde, recon_mask * y0)

            denoi_loss = mse_loss(mask_omega * outputs, mask_omega * (y0 + noise1))
            correction = alpha_weight * sigma1 ** 2 * (torch.sum(y != 0) / torch.numel(y))
            denoi_loss_old.append(denoi_loss * alpha_weight)
            denoi_loss_no_correction.append(denoi_loss * alpha_weight**2)
            denoi_loss_est = denoi_loss * alpha_weight**2 - correction

            denoi_loss_actual = mse_loss(mask_omega * y0_est_tilde, mask_omega * y0 )

            print(denoi_loss_est, correction)
            print(denoi_loss_actual)

            recon_loss = mse_loss(recon_mask * outputs, recon_mask * (y0 + noise1))
            correction = - (torch.sum(recon_mask) / torch.numel(recon_mask)) * sigma1 ** 2
            # correction = - ((torch.sum(y == 0)) / torch.numel(y)) * sigma1 ** 2
            print(recon_loss, 2 * torch.sum(recon_mask * outputs * noise2) / (torch.numel(y) * alpha_sq ), correction)

            total_loss_est = denoi_loss_est + recon_loss + correction

            Ro = torch.sum(y != 0) / torch.numel(y)
            Rpad = torch.sum(pad) / torch.numel(pad)

            # total_correction = alpha_weight * sigma1 ** 2 * (torch.sum(y != 0) / torch.numel(y)) + (torch.sum(recon_mask) / torch.numel(recon_mask)) * sigma1 ** 2
            total_correction = sigma1**2 * (Ro + (1 - Rpad) * alpha_sq) / alpha_sq
            total_loss_est = mse_loss((recon_mask + alpha_weight*mask_omega )* outputs, (recon_mask + alpha_weight*mask_omega) * (y0 + noise1)) \
                               - total_correction

        elif meth == 'n2n_ssdu':
            mask_lo = mask_omega * mask_lambda
            mask_lo[pad_reshaped] = 0
            denoi_loss = mse_loss(mask_lo * outputs, mask_lo * (y0 + noise1))
            correction = alpha_weight * sigma1 ** 2 * (torch.sum(mask_lo) / torch.numel(mask_lo))
            denoi_loss_old.append(denoi_loss * alpha_weight)
            denoi_loss_no_correction.append(denoi_loss * alpha_weight**2)
            denoi_loss_est = denoi_loss * alpha_weight**2 - correction

            denoi_loss_actual = mse_loss(mask_lo * y0_est_tilde, mask_lo * y0)

            # prob_omega = genPDF(config['data']['nx'], config['data']['ny'], 1 / config['data']['us_fac'], config['data']['poly_order'],
            #                          config['data']['fully_samp_size'], config['data']['sample_type'])
            # prob_lambda = genPDF(config['data']['nx'], config['data']['ny'], 1 / config['data']['us_fac_lambda'],
            #                     config['data']['poly_order'], config['data']['fully_samp_size'], config['data']['sample_type'])

            recon_mask = (1 - mask_omega*mask_lambda)
            recon_mask[pad_reshaped] = 0
            recon_loss_actual = mse_loss(recon_mask * y0_est_tilde, recon_mask * y0)

            ssdu_recon_mask = mask_omega * (1 - mask_lambda)
            ssdu_recon_mask[pad_reshaped] = 0
            weighted_mask = ssdu_recon_mask * Kweight

            recon_loss = mse_loss(weighted_mask * outputs, weighted_mask * y)
            #m = weighted_mask ** 2 # / (1 + alpha_sq)
            #correction = - torch.mean(m) * sigma1 ** 2   # * torch.sum(recon_mask) / torch.numel(recon_mask)
            correction =  - (torch.sum(recon_mask) / torch.numel(recon_mask)) * sigma1 ** 2
            print(recon_loss, correction)

            total_loss_est = recon_loss + correction + denoi_loss_est

        denoi_loss_est_all.append(denoi_loss_est)
        denoi_loss_est_actual.append(denoi_loss_actual)
        recon_loss_est_actual.append(recon_loss_actual)

        recon_loss_no_correction.append(recon_loss)
        recon_loss_est_all.append(recon_loss + correction)

        print('True denoi loss is {} and estimate is {}'.format(denoi_loss_actual, denoi_loss_est))
        print('True recon loss is {} and estimate is {}'.format(recon_loss_actual, recon_loss + correction))
        print('True total loss is {} and estimate is {}'.format(total_loss_actual, total_loss_est))


import matplotlib.pyplot as plt

denoi_loss_est_all = np.array(denoi_loss_est_all)
denoi_loss_est_actual = np.array(denoi_loss_est_actual)
denoi_loss_no_correction = np.array(denoi_loss_no_correction)
denoi_loss_old = np.array(denoi_loss_old)

recon_loss_est_all = np.array(recon_loss_est_all)
recon_loss_est_actual = np.array(recon_loss_est_actual)
recon_loss_no_correction = np.array(recon_loss_no_correction)

perc_wrong = lambda x: 100 * (x - denoi_loss_est_actual) / denoi_loss_est_actual

plt.figure(figsize=(9, 3))
plt.subplot(241)
plt.plot(alpha_all, denoi_loss_est_actual)
plt.title('True loss')
plt.xlabel('alpha')
plt.ylim([0.00025, 0.0015])
#plt.ylim([0.00025, 0.0015])
plt.subplot(242)
plt.plot(alpha_all, denoi_loss_old)
plt.title('Estimated loss: old')
plt.ylim([0.00025, 0.0015])
plt.subplot(243)
plt.plot(alpha_all, denoi_loss_no_correction)
plt.title('Estimated loss: new without correction')
plt.subplot(244)
plt.plot(alpha_all, denoi_loss_est_all)
plt.title('Estimated loss with correction')
plt.ylim([0.00025, 0.0015])

plt.subplot(246)
plt.plot(alpha_all, perc_wrong(denoi_loss_old))
plt.title('% wrong')
plt.xlabel('alpha')
plt.subplot(247)
plt.plot(alpha_all, perc_wrong(denoi_loss_no_correction))
plt.title('% wrong')
plt.xlabel('alpha')
plt.subplot(248)
plt.plot(alpha_all, perc_wrong(denoi_loss_est_all))
plt.title('% wrong')
plt.xlabel('alpha')

perc_wrong = lambda x: 100 * (x - recon_loss_est_actual) / recon_loss_est_actual

plt.figure(figsize=(6, 3))
plt.subplot(231)
plt.plot(alpha_all, recon_loss_est_actual)
plt.title('True loss')
plt.xlabel('alpha')
plt.ylim([0.00012, 0.00020])
# plt.ylim([0.00025, 0.0015])
plt.subplot(232)
plt.plot(alpha_all, recon_loss_no_correction)
plt.title('Estimated loss: old (no correction)')
# plt.ylim([0.00025, 0.0015])
plt.subplot(233)
plt.plot(alpha_all, recon_loss_est_all)
plt.ylim([0.00012, 0.00020])
plt.title('Estimated loss: new without correction')


plt.subplot(235)
plt.plot(alpha_all, perc_wrong(recon_loss_no_correction))
plt.title('% wrong')
plt.xlabel('alpha')
plt.subplot(236)
plt.plot(alpha_all, perc_wrong(recon_loss_est_all))
plt.title('% wrong')
plt.xlabel('alpha')

plt.show()