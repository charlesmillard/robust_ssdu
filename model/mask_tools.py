import numpy as np
import torch

import torch.distributions.bernoulli as bern
# from utils_n2n.utils import mse_loss


def gen_pdf(nx, ny, delta, p, c_sq, sample_type):
    if delta == 1.0:
        prob_map = torch.ones((nx, ny))
    else:
        if sample_type == "bern":
            prob_map = gen_pdf_bern(nx, ny, delta, p, c_sq, 0)
        elif sample_type == "bern_inv":
            prob_map = gen_pdf_bern(nx, ny, delta, p, c_sq, 1)
        elif sample_type == "columns":
            prob_map = gen_pdf_columns(nx, ny, delta, p, c_sq)
        else:
            raise Exception("Invalid sample type chosen")
    return prob_map


def gen_pdf_bern(nx, ny, delta, p, c_sq, inv_flag):
    # generate polynomial variable density with sampling factor delta, fully sampled central square c_sq
    if p == 0:
        scale = (delta * nx * ny - c_sq ** 2) / (ny * nx - c_sq ** 2)
        prob_map = torch.ones([nx, ny]) * scale
        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1
    else:
        xv, yv = torch.meshgrid(torch.linspace(-1, 1, ny), torch.linspace(-1, 1, nx), indexing='xy')

        r = torch.sqrt(xv ** 2 + yv ** 2)
        r /= torch.max(r)

        prob_map = (1 - r) ** p
        prob_map[prob_map > 1] = 1
        prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        a = -1
        b = 1

        eta = 1e-3

        ii = 1
        while 1:
            c = a / 2 + b / 2
            prob_map = (1 - r) ** p + c
            prob_map[prob_map > 1] = 1
            prob_map[prob_map < 0] = 0

            if inv_flag:
                prob_map = 1 - prob_map

            prob_map[nx // 2 - c_sq // 2:nx // 2 + c_sq // 2, ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

            delta_current = torch.mean(prob_map)
            if delta > delta_current + eta:
                if inv_flag:
                    b = c
                else:
                    a = c
            elif delta < delta_current - eta:
                if inv_flag:
                    a = c
                else:
                    b = c
            else:
                break

            ii += 1
            if ii == 100:
                print('Careful - genPDF did not converge after 100 iterations')
                break

    return prob_map


def gen_pdf_columns(nx, ny, delta, p, c_sq):
    # generate polynomial variable density with sampling factor delta, fully sampled central square c_sq
    if p == 0:
        scale = (delta * ny - c_sq) / (ny - c_sq)
        prob_map = torch.ones([ny, 1]) * scale
        prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1
    else:
        xv, yv = torch.meshgrid(torch.linspace(-1, 1, 1), torch.linspace(-1, 1, ny), indexing='xy')

        r = torch.abs(yv)
        r /= torch.max(r)

        prob_map = (1 - r) ** p
        prob_map[prob_map > 1] = 1
        prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

        a = -1
        b = 1

        eta = 1e-3

        ii = 1
        while 1:
            c = (a + b) / 2
            prob_map = (1 - r) ** p + c
            prob_map[prob_map > 1] = 1
            prob_map[prob_map < 0] = 0
            prob_map[ny // 2 - c_sq // 2:ny // 2 + c_sq // 2] = 1

            delta_current = torch.mean(prob_map)
            if delta > delta_current + eta:
                a = c
            elif delta < delta_current - eta:
                b = c
            else:
                break

            ii += 1
            if ii == 100:
                print('Careful - genRowsPDF did not converge after 100 iterations')
                break

    prob_map = prob_map.repeat([1, nx])
    prob_map = torch.rot90(prob_map)
    return prob_map


def mask_from_prob(prob_map, sample_type):
    # prob_map[prob_map > 0.99] = 1
    if sample_type in ["bern", "bern_inv"]:
        mask = bern.Bernoulli(prob_map).sample()
    elif sample_type == "columns":
        (nx, ny) = prob_map.shape
        b = bern.Bernoulli(prob_map[0:1])
        mask1d = b.sample()
        mask = mask1d.repeat([nx, 1])

    return mask.unsqueeze(0).unsqueeze(0).float()


def diff_mask_from_prob(prob_map, sample_type):
    # prob_map[prob_map > 0.99] = 1
    if sample_type in ["bern", "bern_inv"]:
        mask = bern.Bernoulli(prob_map).sample()
    elif sample_type == "columns":
        (nx, ny) = prob_map.shape
        r = torch.distributions.uniform.Uniform(torch.zeros((1, ny)), torch.ones((1, ny)))
        mask1d = 1 / (1 + torch.exp(- 20 * (prob_map[0:1] - r.rsample())))
        mask = mask1d.repeat([nx, 1])

    return mask.unsqueeze(0).unsqueeze(0).float()


def gen_kweight(prob_omega, prob_lambda):
    kweight = (1 - prob_lambda * prob_omega) / (prob_omega * (1 - prob_lambda))
    kweight[torch.isnan(kweight)] = 1e-5
    kweight[torch.isinf(kweight)] = 1e-5
    return kweight ** 0.5


def mask_corners(nx, ny, sq_sz):
    backg_mask = torch.zeros((nx, ny))
    backg_mask[:sq_sz, :sq_sz] = 1
    backg_mask[-sq_sz:, :sq_sz] = 1
    backg_mask[:sq_sz, -sq_sz:] = 1
    backg_mask[-sq_sz:, -sq_sz:] = 1
    return backg_mask


def make_lambda_mask(prob_omega, lambda_scale, config):
    prob_lambda = gen_pdf(config['data']['nx'], config['data']['ny'], 1 / lambda_scale,
                          config['mask']['poly_order_lambda'], config['mask']['fully_samp_size_lambda'],
                          config['mask']['sample_type_lambda'])
    mask_lambda = torch.zeros(prob_omega.shape).unsqueeze(1).unsqueeze(1)
    for ii in range(mask_lambda.shape[0]):
        m2 = mask_from_prob(prob_lambda, config['mask']['sample_type_lambda'])
        #m2 = diffMaskFromProb(prob_lambda, config['data']['sample_type'])
        mask_lambda[ii] = m2

    kweight = gen_kweight(prob_omega, prob_lambda.unsqueeze(0).repeat([prob_omega.shape[0], 1, 1]).to(prob_omega))

    return mask_lambda, kweight
