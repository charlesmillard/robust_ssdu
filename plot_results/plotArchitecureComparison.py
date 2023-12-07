import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.preparations import load_config
import matplotlib.patches as mpatches

import matplotlib.ticker as ticker

rt = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/'
roots = ['architectures', 'whole_experiment']

loss_type = 'loss'

log_loc = []
for r in roots:
    for f in os.walk(rt + r):
        if 'state_dict' in f[2]:
            log_loc.append(f[0])

sigs = [0.02, 0.04, 0.06, 0.08]
us_facs = [4, 8]

denoi_mods = ['unet_single', 'unet']

nm = ['Standard VarNet', 'Denoising VarNet']

col = ['#CC2200', 'green']
markers = ['x', '+',]
linestyles = ['-.', ':']

all_nmse = np.zeros((len(denoi_mods), len(sigs), len(us_facs)))

ii = 0
for log in log_loc:
    print(log)
    config = load_config(log + '/config')

    arch = config['network']['denoi_model']
    sigma1 = config['data']['sigma1']
    meth = config['data']['method']
    us_fac = config['data']['us_fac']

    if meth == 'full':
        arch_idx = denoi_mods.index(arch) if arch in denoi_mods else 'none'
        sig_idx = sigs.index(sigma1) if sigma1 in sigs else 'none'
        us_fac_idx = us_facs.index(us_fac) if us_fac in us_facs else 'none'
    else:
        arch_idx, sig_idx, us_fac_idx = ['none', 'none', 'none']

    if os.path.exists(log + '/results.npz') and 'none' not in [arch_idx, sig_idx, us_fac_idx]:
        res = np.load(log + '/results.npz')
        if all_nmse[arch_idx, sig_idx, us_fac_idx] == 0:
            all_nmse[arch_idx, sig_idx, us_fac_idx]= np.mean(res[loss_type])
        else:
            print('double attempted at ({}, {}), file {}'.format(arch_idx, sig_idx, us_fac_idx, log))

    ii += 1

all_nmse[all_nmse == 0] = None
print(all_nmse)
sns.set_theme()

baseline_res = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984/results.npz'
baseline_nmse = np.mean(np.load(baseline_res)[loss_type])

fig = plt.figure(figsize=(6, 3))
plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

ax = fig.add_subplot(121)
# ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.02))

for ii in range(2):
    ax.plot(sigs, all_nmse[ii, :, 0], col[ii], marker=markers[ii],  linestyle=linestyles[ii])

ax.set_ylabel('Test set NMSE',  fontsize=14)
ax.ticklabel_format(axis='y', style='scientific', useMathText=True)
# ax.set_xlim([0.015, 0.085])
ax.set_ylim([0.00015, 0.0011])
ax.set_xlabel(r'Measurement noise $\sigma_n$',  fontsize=14)
ax.set_title('$R_\Omega=4$')

plt.legend(nm, ncol=1, fontsize=10, loc='upper left')
fig.tight_layout()


# plt.add_subplot(122)
#
# plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
# plt.rc('axes', labelsize=20)
# plt.rc('xtick', labelsize=12)
# plt.rc('ytick', labelsize=12)

ax2 = fig.add_subplot(122)
#ax2 = plt.axes()
ax2.xaxis.set_major_locator(ticker.MultipleLocator(0.02))

for ii in range(2):
    ax2.plot(sigs, all_nmse[ii, :, 1], col[ii], marker=markers[ii],  linestyle=linestyles[ii])

ax2.set_ylabel('Test set NMSE',  fontsize=14)
# ax.set_xlim([0.015, 0.085])
#ax.set_ylim([-5, 100])
ax2.set_xlabel(r'Measurement noise $\sigma_n$',  fontsize=14)
ax2.set_title('$R_\Omega=8$')
fig.tight_layout()

plt.show()


