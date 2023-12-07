import torch
import torch.distributions as dist

from utils.mask_tools import *

nx = 100
ny = 200

prob_omega = gen_pdf(nx, ny, 2, 0, 10, 'bern')



p = torch.tensor([0.2])
n_epoch = 10

for epoch in range(n_epoch):

    m = dist.Bernoulli(p)
    action = m.sample()
    loss = - m.log_prob(action)
    loss.backward()

    print(m.log_prob(action))