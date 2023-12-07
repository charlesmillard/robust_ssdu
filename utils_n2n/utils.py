import numpy as np
import torch
import torch.nn.functional as F
import os

from fastmri.fftc import fft2c_new, ifft2c_new
from scipy.linalg import sqrtm, inv

from model.mask_tools import mask_corners

import warnings
import h5py
from datetime import datetime


def rmse_loss(xhat, x0):
    return torch.sqrt(torch.mean(torch.abs(xhat - x0) ** 2))


def nmse_loss(xhat, x0):
    return torch.mean(torch.abs(xhat - x0) ** 2) / torch.mean(torch.abs(x0) ** 2)


def mse_loss(xhat, x0):
    return torch.mean(torch.abs(xhat - x0) ** 2)


def l12loss(xhat, x0):
    l2 = torch.linalg.vector_norm((xhat - x0), 2) / torch.linalg.vector_norm(x0, 2)
    l1 = torch.linalg.vector_norm((xhat - x0), 1) / torch.linalg.vector_norm(x0, 1)
    return (l1 + l2) / 2


def image_mag_loss(xhat, x0):
    xhat_crop = kspace_to_rss(xhat)
    x0_crop = kspace_to_rss(x0)

    loss = torch.mean((xhat_crop - x0_crop) ** 2) / torch.mean(x0_crop ** 2) \
           + torch.mean(torch.abs(xhat_crop - x0_crop)) / torch.mean(torch.abs(x0_crop))

    return loss / 2


def nmse(x, x0):
    return np.mean(np.abs(x - x0) ** 2) / np.mean(np.abs(x0) ** 2)


def pad_or_trim_tensor(x, nx_target, ny_target):
    _, _, nx, ny = x.shape
    if nx != nx_target or ny != ny_target:
        x = kspace_to_im(torch.unsqueeze(x, 0))
        dx = nx_target - nx
        if dx > 0:
            x = F.pad(x, (0, 0, dx // 2, dx // 2 + dx % 2))
        elif dx < -1:
            x = x[:, :, :, -dx // 2:(dx + 2) // 2 - 1]
        elif dx == -1:
            x = x[:, :, :-1]

        dy = ny_target - ny
        if dy > 0:
            x = F.pad(x, (dy // 2, dy // 2 + dy % 2))
        elif dy < -1:
            x = x[:, :, :, :, -dy // 2:(dy + 2) // 2 - 1]
        elif dy == -1:
            x = x[:, :, -1]

        x = im_to_kspace(x)
    return torch.squeeze(x)


def kspace_to_im(y):
    y = torch.permute(y, (0, 2, 3, 4, 1)).contiguous()
    x = ifft2c_new(y)
    x = torch.permute(x, (0, 4, 1, 2, 3))
    return x


def im_to_kspace(x):
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous()
    y = fft2c_new(x)
    y = torch.permute(y, (0, 4, 1, 2, 3))
    return y


def pass_unet(y, base_net):
    x = kspace_to_im(y)
    (n1, n2, n3, n4, n5) = x.shape
    x = torch.reshape(x, (n1, n2 * n3, n4, n5))
    outputs = base_net(x)
    outputs = torch.reshape(outputs, (n1, n2, n3, n4, n5))
    outputs = im_to_kspace(outputs)
    return outputs


def pass_varnet(y, base_net):
    y_tild = torch.permute(y, (0, 2, 3, 4, 1))
    outputs = base_net(y_tild, (y_tild != 0).bool())
    outputs = torch.permute(outputs, (0, 4, 1, 2, 3))
    return outputs


def kspace_to_rss(y):
    x = kspace_to_im(y)
    return im_to_rss(x)


def im_to_rss(x):
    x = torch.permute(x, (0, 2, 3, 4, 1)).contiguous()
    x = torch.view_as_complex(x)
    x = torch.sqrt(torch.sum(torch.abs(x) ** 2, 1))
    (_, nx, ny) = x.shape
    crop_sz = min([160, nx // 2])
    x = x[:, nx // 2 - crop_sz:nx // 2 + crop_sz]
    return x.unsqueeze(1)


def compute_number_of_params(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def sure_loss(net, y, y_est, mask, tau, sigma):
    y = y.to(y_est)
    mask = mask.to(y_est)

    noise_var = (sigma ** 2).to(y)
    tau = (torch.max(torch.abs(y)) * 10 ** -6).to(y)
    tau = tau.to(y)
    # generate a random vector b

    b = torch.randn_like(y).to(y)
    y2 = mask * net(mask * (y + tau * b))
    y2 = y2.to(y_est)

    # compute batch size K
    K = torch.tensor(y.shape[0]).to(y_est)

    # compute n (dimension of x)
    n = (torch.numel(y) / K).to(y_est)
    m = (n * torch.mean(mask)).to(y_est)

    n_mask = torch.numel(mask) * K  # .to(y_est)

    # compute loss_sure
    # loss_sure = torch.sum(mask * (y_est - y).pow(2)) / (K * m) - torch.sum(mask) * noise_var / (K * m) \
    #             + (2 * noise_var / (tau * m * K)) * (b * mask * (y2 - y_est)).sum()
    loss_sure = torch.mean(mask * (y_est - y).pow(2)) - torch.mean(mask) * noise_var \
                + (2 * noise_var / tau) * (b * mask * (y2 - y_est)).mean()

    # loss_sure *= K * m
    # loss_sure /= torch.sum(torch.abs(mask * y) ** 2)

    if torch.isnan(loss_sure):
        loss_sure = torch.tensor(0.0)
        warnings.warn("SURE loss found to be NaN - setting to zero")
        # raise Exception('loss_sure is nan')

    return loss_sure


def data_root(config):
    if config['data']['set'] == 'fastmri':
        if os.path.isdir('/home/fs0/xsd618/'):  # jalapeno
            root = '/home/fs0/xsd618/scratch/fastMRI_brain/multicoil_'
        elif os.path.isdir('/well/chiew/users/fjv353/'):  # rescomp
            root = '/well/chiew/users/fjv353/fastMRI_brain/multicoil_'
        else:  # my laptop
            root = '/home/xsd618/data/fastMRI_test_subset_brain/multicoil_'
    elif config['data']['set'] == 'm4raw':
        if os.path.isdir('/home/fs0/xsd618/'):  # jalapeno
            root = ''
        elif os.path.isdir('/well/chiew/users/fjv353/'):  # rescomp
            root = '/gpfs3/well/chiew/users/fjv353/m4raw/multicoil_'
        else:  # my laptop
            root = '/home/xsd618/data/m4raw/multicoil_'
    else:
        raise Exception("Invalaid dataset selected")

    return root


def m4raw_averages(f, s, config):
    root = data_root(config) + 'test/'

    file_list = os.listdir(root)

    f_rt = f

    backg_mask = torch.zeros((config['data']['nx'], config['data']['ny']))
    sq_sz = 30
    backg_mask[:sq_sz, :sq_sz] = 1
    backg_mask[-sq_sz:, :sq_sz] = 1
    backg_mask[:sq_sz, -sq_sz:] = 1
    backg_mask[-sq_sz:, -sq_sz:] = 1

    y0 = torch.as_tensor(h5py.File(root + f, 'r')['kspace'][s])
    y0 = torch.flip(y0, [1])
    y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))
    y0 = pad_or_trim_tensor(y0, config['data']['nx'], config['data']['ny'])

    is_colored = True

    sig_inv = whitening_mtx(y0, backg_mask, is_colored)

    x0_estimates = 0
    n_est = 0
    for file_idx in range(len(file_list)):
        file = file_list[file_idx]
        if file[:-5] == f_rt[:-5] and file != f_rt:
            y0 = torch.as_tensor(h5py.File(root + file, 'r')['kspace'][s])
            y0 = torch.flip(y0, [1])
            y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))
            y0 = pad_or_trim_tensor(y0, config['data']['nx'], config['data']['ny'])
            y0 = whiten_kspace(y0, sig_inv)
            x0_estimates += kspace_to_rss(y0.unsqueeze(0))
            n_est += 1

    print('Number of averages is {}'.format(n_est))

    x0 = x0_estimates / n_est

    return x0.unsqueeze(0).float()


def whitening_mtx(y0, backg_mask, is_colored):
    x = kspace_to_im(torch.unsqueeze(y0, 0))

    _, _, nc, nx, ny = x.shape
    n_backg = 2 * np.sum(backg_mask.numpy())
    all_noise = np.zeros((nc, int(n_backg)))
    for c in range(nc):
        all_noise[c, :] = x[:, :, c][
            backg_mask.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1) != 0]  # x_coil[x_coil != 0]

    if is_colored:
        noise_cov = np.cov(all_noise)
        try:
            sig_inv = sqrtm(inv(noise_cov))
        except:
            sig_inv = 1 / np.std(all_noise)
    else:
        sig_inv = 1 / np.std(all_noise)

    return sig_inv


def whiten_kspace(y, sig_inv):
    if np.size(sig_inv) > 1:
        _, nc, nx, ny = y.shape
        y0_reshape = torch.reshape(y.permute((1, 0, 2, 3)), (nc, ny * nx * 2))
        y0_white = np.matmul(sig_inv, y0_reshape)
        y0_white = torch.reshape(y0_white, (nc, 2, nx, ny))
        y = y0_white.permute((1, 0, 2, 3))
    else:
        y = y * sig_inv
    return y


def unwhiten_kspace(f, s, y, config):
    root = data_root(config) + 'test/'

    y0 = torch.as_tensor(h5py.File(root + f, 'r')['kspace'][s])
    y0 = torch.flip(y0, [1])
    y0 = torch.permute(torch.view_as_real(y0), (3, 0, 1, 2))
    y0 = pad_or_trim_tensor(y0, config['data']['nx'], config['data']['ny'])

    backg_mask = mask_corners(config['data']['nx'], config['data']['ny'], 30)

    is_colored = True

    sig_inv = whitening_mtx(y0, backg_mask, is_colored)
    sig = inv(sig_inv)

    return whiten_kspace(y, sig)


def zero_kspace(k, zero_loc):
    if k.shape[1] == 1:
        zero_loc = zero_loc[:, 0:1, 0:1]

    zero_loc = zero_loc.to(k).bool()
    if k.type() == "torch.BoolTensor":
        k[zero_loc] = False
    else:
        k[zero_loc] = 0

    return k


def set_seeds(seed):
    if seed is None:
        # millisecond precision
        _, seed = repr(datetime.now().timestamp()).split('.')
        seed = int(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return 0