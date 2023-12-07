import torchvision

from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from utils.preparations import *


config = load_config('logs/cuda/8x/full/76846908_0.04/config')
config['network']['device'] = 'cpu'
test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=True)

torch.manual_seed(540)
sigma1 = float(config['data']['sigma1'])
with torch.no_grad():
    for i, data in enumerate(test_load, 0):
        y0, noise1, noise2, mask_omega, mask_lambda, mu, mask = data
        noise1 *= sigma1
        x0 = kspace_to_rss(y0)
        x_noisy = kspace_to_rss(y0 + noise1)
        x_noisy_sub = kspace_to_rss(mask_omega * (y0 + noise1))

        break

def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4).detach().cpu()
    torchvision.utils.save_image(im, name)


mx = torch.max(torch.abs(x0[0]))
saveIm(x0/ mx, 1, 'figs/training_data_examples/x0.png')
saveIm(x_noisy/ mx, 1, 'figs/training_data_examples/x_noisy.png')
saveIm(x_noisy_sub/ mx, 1, 'figs/training_data_examples/x_noisy_sub.png')