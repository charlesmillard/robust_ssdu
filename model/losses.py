from utils_n2n.utils import *
from model.mask_tools import *


def training_loss(data, net, config, criterion, lambda_scale, alpha_param):
    meth = config['optimizer']['method']
    dev = config['network']['device']

    alpha = alpha_param[0]
    alpha_sq = alpha ** 2
    alpha_weight = (1 + alpha_sq) / alpha_sq

    if config['noise']['sim_noise']:
        sigma1 = torch.tensor(float(config['noise']['sigma1']))
    else:
        sigma1 = torch.tensor(1)
    sigma2 = alpha * sigma1

    y0, noise1, mask_omega, prob_omega, mask_im, slice_info = data

    set_seeds(None)
    mask_lambda, kweight = make_lambda_mask(prob_omega, lambda_scale, config)
    noise2 = torch.randn(noise1.shape)

    def zero_k(k):
        return zero_kspace(k, torch.abs(y0) == 0)

    noise1 = zero_k(noise1)
    noise2 = zero_k(noise2)

    mask_omega = zero_k(mask_omega)
    mask_lambda = zero_k(mask_lambda)

    kweight = kweight[0]
    noise1 *= sigma1
    noise2 *= sigma2

    if not config['noise']['sim_noise']:
        noise1 = 0

    y_noisy = y0 + noise1
    y_target = y0 if meth == 'full' else y_noisy

    y_noisy = y_noisy.to(dev)
    mask_lambda = mask_lambda.to(dev).bool()
    mask_omega = mask_omega.to(dev).bool()
    noise2 = noise2.to(dev)
    kweight = kweight.to(dev)
    alpha_weight = alpha_weight.to(dev)

    if meth in ['full', 'noise2full']:
        y_input = mask_omega * y_noisy
        mask_target = torch.ones(mask_omega.shape).to(dev)
    elif meth in ['ssdu', 'noise2recon']:
        y_input = mask_lambda * mask_omega * y_noisy
        mask_target = mask_omega * ~mask_lambda
        if config['optimizer']['K_weight']:
            mask_target = kweight * mask_target.float()
    elif meth == 'noisier2full':
        y_input = mask_omega * (y_noisy + noise2)
        if config['optimizer']['alpha_weight']:
            mask_target = (~mask_omega).float() + alpha_weight * mask_omega.float()
        else:
            mask_target = torch.ones(mask_omega.shape)
    elif meth == 'robust_ssdu':
        y_input = mask_lambda * mask_omega * (y_noisy + noise2)
        recon_mask = mask_omega * ~mask_lambda
        if config['optimizer']['K_weight']:
            recon_mask = kweight * recon_mask.float()
        denoi_mask = mask_omega * mask_lambda
        if config['optimizer']['alpha_weight']:
            # denoi_mask = alpha_weight ** 0.5 * denoi_mask.float()
            denoi_mask = alpha_weight * denoi_mask.float()
        mask_target = recon_mask + denoi_mask
    else:
        raise Exception('Invalid training method chosen')

    mask_target = zero_k(mask_target)

    outputs = net(y_input).to(dev)

    # compute loss
    loss = criterion(mask_target * outputs, mask_target * y_target)
    if meth == 'noise2recon':
        y_input_denoi = mask_omega * (y_noisy + noise2)
        outputs_denoi = net(y_input_denoi).to(dev)
        denoi_loss = criterion(zero_k(outputs_denoi), zero_k(outputs))
        loss = config['optimizer']['noise2recon_lamb'] * loss + denoi_loss

    # compute estimates
    if meth in ['full', 'noise2full', 'noise2recon']:
        y0_est = outputs
    elif meth == 'ssdu':
        y0_est = ~mask_omega * outputs + mask_omega * y_noisy
    elif meth == 'noisier2full':
        y0_est = mask_omega * ((1 + alpha_sq) * outputs - y_input) / alpha_sq + ~mask_omega * outputs
    elif meth == 'robust_ssdu':
        denoi_mask = mask_omega * mask_lambda
        y0_est = denoi_mask * ((1 + alpha_sq) * outputs - y_input) / alpha_sq + ~denoi_mask * outputs

    y0_est = zero_k(y0_est)

    return loss, y0_est
