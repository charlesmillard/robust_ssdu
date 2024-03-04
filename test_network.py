from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from skimage.metrics import structural_similarity as ssim
from utils_n2n.preparations import *
from model.mask_tools import *
from configs.config_loader import *

from packages.bm3d import bm3d

import os
import time
import argparse


def main(config, log_loc):
    with torch.no_grad():
        batch_sz = 10 if torch.cuda.is_available() else 2
        test_load = DataLoader(ZfData('test', config), batch_size=batch_sz, shuffle=True)

        pass_network, network = create_network(config)

        alpha = torch.tensor(config['noise']['alpha'])
        alpha_sq = alpha ** 2

        if config['noise']['sim_noise']:
            sigma1 = torch.tensor(float(config['noise']['sigma1']))
        else:
            sigma1 = torch.tensor(1)

        sigma2 = alpha * sigma1

        meth = config['optimizer']['method']
        dev = config['network']['device']

        network.load_state_dict(torch.load(log_loc + '/state_dict', map_location=config['network']['device']))
        network.to(dev)
        network.eval()

        loss, loss_tilde = [], []
        all_ssim, all_ssim_tilde, all_ssim_bm3d = [], [], []
        loss_rss, loss_rss_tilde, loss_rss_bm3d = [], [], []
        time_all, time_all_bm3d = [], []

        for i, data in enumerate(test_load, 0):
            y0, noise1, mask_omega, prob_omega, _, slice_info = data

            nch = y0.shape[0]
            # only evaluate performance on one of m4raw's repetitions
            if config['data']['set'] == 'm4raw':
                idx_of_zeros = []
                for ii in range(nch):
                    if slice_info[0][ii][-5:] == '01.h5':
                        idx_of_zeros.append(ii)
                nch = len(idx_of_zeros)

            if nch > 0:
                if config['data']['set'] == 'm4raw':
                    y0 = y0[idx_of_zeros]
                    noise1 = noise1[idx_of_zeros]
                    mask_omega = mask_omega[idx_of_zeros]

                    slice_info_new = ([], [])
                    for ii in range(nch):
                        slice_info_new[0].append(slice_info[0][ii])
                        slice_info_new[1].append(slice_info[1][ii])
                    slice_info = slice_info_new

                set_seeds(i)
                mask_lambda, _ = make_lambda_mask(prob_omega, config['mask']['us_fac_lambda'], config)

                y0 = y0.to(dev)
                mask_lambda = mask_lambda.to(dev).bool()
                mask_omega = mask_omega.to(dev).bool()
                noise1 = noise1.to(dev)
                noise2 = torch.randn(y0.shape).to(dev)

                zero_k = lambda k: zero_kspace(k, torch.abs(y0) == 0)

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

                x0 = kspace_to_rss(y0)

                if config['data']['set'] == 'm4raw':
                    for d in range(nch):
                        x0[d] = m4raw_averages(slice_info[0][d], int(slice_info[1][d]), config)

                if meth in ['full', 'noise2full', 'noise2recon']:
                    t1 = time.time()
                    outputs = pass_network(y, network)
                    t2 = time.time()
                    y0_est = outputs
                    y0_est_tilde = outputs
                elif meth == 'ssdu':
                    t1 = time.time()
                    y0_est = pass_network(y, network)
                    y0_est = y0_est * (y == 0) + y * (y != 0)
                    t2 = time.time()
                    y0_est_tilde = pass_network(mask_lambda * y, network)
                    y0_est_tilde = y0_est_tilde * (y == 0) + y * (y != 0)
                elif meth == 'noisier2full':
                    t1 = time.time()
                    outputs = pass_network(y, network)
                    y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)
                    t2 = time.time()

                    y_tilde = y + mask_omega * noise2
                    outputs = pass_network(y_tilde, network)
                    y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
                elif meth == 'robust_ssdu':
                    t1 = time.time()
                    outputs = pass_network(y, network)
                    y0_est = (y != 0) * ((1 + alpha_sq) * outputs - y) / alpha_sq + outputs * (y == 0)
                    t2 = time.time()

                    y_tilde = mask_lambda*(y + mask_omega * noise2)
                    outputs = pass_network(y_tilde, network)
                    y0_est_tilde = (y_tilde != 0) * ((1 + alpha_sq) * outputs - y_tilde) / alpha_sq + outputs * (y_tilde == 0)
                elif meth == "r2r_ssdu":
                    t1 = time.time()
                    outputs = pass_network(y, network)
                    y0_est = (y != 0) * outputs + outputs * (y == 0)
                    t2 = time.time()

                    y_tilde = mask_lambda * (y + mask_omega * noise2)
                    outputs = pass_network(y_tilde, network)
                    y0_est_tilde = (y_tilde != 0) * outputs + outputs * (y_tilde == 0)

                y0_est = zero_k(y0_est)
                y0_est_tilde = zero_k(y0_est_tilde)

                pad_mask = x0 > 0

                x0_est = kspace_to_rss(y0_est)
                x0_est_tilde = kspace_to_rss(y0_est_tilde)

                x0_est = x0_est * pad_mask
                x0_est_tilde = x0_est_tilde * pad_mask
                x0 = x0 * pad_mask

                x0_est = x0_est.to(x0)

                for ii in range(nch):
                    if meth == 'ssdu':
                        t3 = time.time()
                        x0_bm3d = bm3d(x0_est[ii, 0].cpu(), sigma_psd=torch.std(x0_est - x0).cpu())
                        x0_bm3d = torch.as_tensor(x0_bm3d).unsqueeze(0).unsqueeze(0)
                        t4 = time.time()
                        time_all_bm3d.append((t2 - t1) / nch + (t4 - t3))
                    else:
                        x0_bm3d = torch.as_tensor(x0_est[ii, 0]).unsqueeze(0).unsqueeze(0)
                        time_all_bm3d.append(0)

                    time_all.append((t2 - t1) / nch)

                    x1 = np.array((x0_est[ii, 0]).detach().cpu())
                    x1_tilde = np.array((x0_est_tilde[ii, 0]).detach().cpu())
                    x1_bm3d = np.array((x0_bm3d[0, 0]).detach().cpu())
                    x2 = np.array((x0[ii, 0]).detach().cpu())
                    all_ssim.append(ssim(x1, x2, data_range=x2.max(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False))
                    loss_rss.append(nmse(x1, x2))

                    all_ssim_tilde.append(ssim(x1_tilde, x2, data_range=x2.max()))
                    loss_rss_tilde.append(nmse(x1_tilde, x2))

                    all_ssim_bm3d.append(ssim(x1_bm3d, x2, data_range=x2.max()))
                    loss_rss_bm3d.append(nmse(x1_bm3d, x2))

                    y1 = y0_est[ii].detach().cpu()
                    y1_tilde = y0_est_tilde[ii].detach().cpu()
                    y2 = y0[ii].detach().cpu()
                    loss.append(mse_loss(y1, y2))
                    loss_tilde.append(mse_loss(y1_tilde, y2))

                np.savez(log_loc + '/results.npz', loss=loss, loss_tilde=loss_tilde,
                         loss_rss=loss_rss, loss_rss_tilde=loss_rss_tilde,
                         all_ssim=all_ssim, all_ssim_tilde=all_ssim_tilde,
                         all_ssim_bm3d=all_ssim_bm3d, loss_rss_bm3d=loss_rss_bm3d,
                         time_all_bm3d=time_all_bm3d, time_all=time_all)

            if not torch.cuda.is_available():
                break

    if not torch.cuda.is_available():
        log_loc = log_loc + '/cpu_examples'
        if not os.path.isdir(log_loc):
            os.mkdir(log_loc)

    f = open(log_loc + '/test_results.txt', 'w')
    f.write('model location is ' + log_loc + '\n')
    f.write('loss: {:e} \n'.format(np.mean(loss).item()))
    f.write('loss (dB): {:e} \n'.format(10*np.log10(np.mean(loss).item())))
    f.write('loss tilde: {:e} \n'.format(np.mean(loss_tilde).item()))
    f.write('RSS loss: {:e} \n'.format(10*np.log10(np.mean(loss_rss).item())))
    f.write('RSS loss tilde: {:e} \n'.format(10*np.log10(np.mean(loss_rss_tilde).item())))
    f.write('RSS loss BM3D: {:e} \n'.format(10*np.log10(np.mean(loss_rss_bm3d).item())))
    f.write('SSIM: {:e} \n'.format(np.mean(all_ssim)))
    f.write('SSIM tilde: {:e} \n'.format(np.mean(all_ssim_tilde)))
    f.write('SSIM BM3D: {:e} \n'.format(np.mean(all_ssim_bm3d)))
    f.write('Time: {:e} \n'.format(np.mean(time_all)))
    f.write('Time BM3D: {:e} \n'.format(np.mean(time_all_bm3d)))


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Robust SSDU')
    args.add_argument('-l', '--log_loc', default='saved/logs', type=str,
                      help='file path to save logs (default: saved/logs)')
    args.add_argument('-d', '--data_loc', default='/home/xsd618/data/fastMRI_test_subset_brain/', type=str,
                      help='data location (default: /home/xsd618/data/fastMRI_test_subset_brain/)')
    args = args.parse_args()

    config_main = load_config(args.log_loc + '/config')
    # config_main = reformat_config(config_main)
    config_main = set_missing_config_entries(config_main)
    config_main = prepare_config(config_main, args)
    config_main['network']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    set_seeds(config_main['optimizer']['seed'])

    main(config_main, args.log_loc)
