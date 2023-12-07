import torch


def sure_loss(denoi, y, y_est, mask, sigma):
    #  estimates the error of y_est, where y is the input to a
    #  denoiser denoi and sigma is the (scalar) noise standard deviation

    noise_var = sigma ** 2
    tau = torch.max(torch.abs(y)) * 10**-6
    b = torch.randn_like(y)

    y2 = mask * denoi(y + tau * b)

    # compute SURE
    loss_sure = torch.mean(mask * (y_est - y).pow(2)) - torch.mean(mask) * noise_var  \
                + (2 * b * mask * noise_var / tau) * torch.mean(y2 - y_est)

    return loss_sure
