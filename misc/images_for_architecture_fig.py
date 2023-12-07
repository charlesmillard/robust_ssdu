import torchvision

from torch.utils.data import DataLoader
from data_loader.zf_data_loader import ZfData

from utils.preparations import *


l = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/4x/0.08/full.yaml/20649536/'
config = load_config(l + 'config')
config['network']['device'] = 'cpu'
test_load = DataLoader(ZfData('test', config), batch_size=1, shuffle=True)

pass_network, network = create_network(config)

network.load_state_dict(torch.load(l + '/state_dict', map_location=config['network']['device']))
network.to(config['network']['device'])
network.eval()

torch.manual_seed(540)
sigma1 = float(config['data']['sigma1'])
with torch.no_grad():
    for i, data in enumerate(test_load, 0):
        y0, noise1, noise2, mask_omega, mask_lambda, mu, mask = data
        noise1 *= sigma1

        y = mask_omega * (y0 + noise1)

        outputs = pass_network(y, network)

        break

def saveIm(im, ndisp, name):
    im = torch.abs(im[0:ndisp])
    im = torchvision.utils.make_grid(im, nrow=4).detach().cpu()
    torchvision.utils.save_image(im, name)

def mag_crop(y):
    return (y[:,0,0, 220:420, 60:260]**2 + y[:,1,0, 220:420, 60:260]**2)**0.1

save_root = '/home/xsd618/noisier2noise_kspace_denoising/figs/architecture_fig/'

y_input = outputs

saveIm(mag_crop(y0), 1, save_root + 'y0.png')
saveIm(mag_crop(y_input), 1, save_root + 'y_input.png')
saveIm(mag_crop(mask_omega * y0), 1, save_root + 'omega_y_input.png')
saveIm(mag_crop((1- mask_omega) * y0), 1, save_root + '1m_omega_y_input.png')

print(mask_omega.shape)
om = mask_omega[:, :, :, 220:420, 60:260] * 1.00

saveIm(om, 1, save_root + 'omega.png')
saveIm(1 - om, 1, save_root + '1m_omega.png')