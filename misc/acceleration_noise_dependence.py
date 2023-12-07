from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from utils.preparations import *

from utils.mask_tools import gen_pdf, mask_from_prob

import matplotlib.pyplot as plt

config = load_config('/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/4x/0.08/full.yaml/17481871/config')
config['network']['device'] = 'cpu'

torch.manual_seed(550)
noise_fix = torch.randn((1, 2, 16, 640, 320)).float()

sigmas = np.linspace(0.0, 0.08, 9)
accels = np.linspace(1.5, 10, 19)

test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=True)
for i, data in enumerate(test_load, 0):
    y0, noise1, noise2, mask_omega, mask_lambda, mu, mask = data

print(torch.mean(torch.abs(y0)), torch.max(torch.abs(y0)))

losses = []
losses_full = []
all_x_full_noisy = []
all_x_noisy = []
sig_idx = 0
with torch.no_grad():
    for a in accels:
        P = gen_pdf(640, 320, 1 / a, 8, 10, 'bern')
        mask_omega = mask_from_prob(P, 'bern')
        #mask_omega = P.copy()
        print(np.mean(mask_omega))
        mask_omega = torch.as_tensor(mask_omega).unsqueeze(0).unsqueeze(0).int()
        losses.append([])
        losses_full.append([])
        all_x_full_noisy.append([])
        all_x_noisy.append([])
        for sigma1 in sigmas:
            noise = noise1 * sigma1

            y_full_noisy = y0 + noise
            y_noisy = mask_omega * y_full_noisy

            all_x_full_noisy[sig_idx].append(kspace_to_rss(y_full_noisy)[0,0])
            all_x_noisy[sig_idx].append(kspace_to_rss(y_noisy)[0,0])

            losses[sig_idx].append(mse_loss(y0, y_noisy))
            losses_full[sig_idx].append(mse_loss(y0, y_full_noisy))

        sig_idx += 1

print(losses)

plt.figure()
plt.plot(accels, np.log(losses))
plt.legend(sigmas, ncol=2, loc='lower right')
plt.xlabel('acceleration')
plt.ylabel('loss')

a_idx = 5
sig_idx = 5

print('losses are {} and {} respectively'.format(losses_full[a_idx][sig_idx], losses[a_idx][sig_idx]))

mx = torch.max(all_x_full_noisy[a_idx][sig_idx])
plt.figure()
plt.subplot(121)
plt.imshow(all_x_full_noisy[a_idx][sig_idx], vmax=mx)
plt.subplot(122)
plt.imshow(all_x_noisy[a_idx][sig_idx], vmax=mx)

plt.show()