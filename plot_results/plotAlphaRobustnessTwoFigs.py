import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.preparations import load_config
import matplotlib.patches as mpatches

import matplotlib.ticker as ticker

rt = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/'
roots = ['alpha_robustness', 'alpha_robustness_n2n_ssdu', 'alpha_robustness_noise2recon',
         'whole_experiment/8x/0.06/', 'weighted_experiment/8x/0.06/',
         'whole_experiment_alph1/8x/0.06', 'whole_experiment_alph1_weighted/8x/0.06']
         # '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/weighted_experiment/8x/0.06/noise2recon.yaml',
         # '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/8x/0.06/noise2recon.yaml']

loss_type = 'loss'

log_loc = []
for r in roots:
    for f in os.walk(rt + r):
        if 'state_dict' in f[2]:
            log_loc.append(f[0])

alphas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

# meths = ['n2n', 'n2n']
# is_weighted = [False, True]
#
meths = ['n2n', 'n2n', 'noise2recon', 'n2n_ssdu',  'n2n_ssdu']
is_weighted = [False, True, None, False,  True]

# meths = ['n2n', 'n2n', 'n2n_ssdu',  'n2n_ssdu']
# is_weighted = [False, True, False,  True]

nm = ['Noisier2Full', 'Weighted Noisier2Full', 'Noise2Recon',
      'Robust SSDU', 'Weighted Robust SSDU']


col = ['black',  'black', 'red', 'green', 'orange', 'blue', 'green', 'red']
col = ['#CC2200', 'green', 'blue', 'black', 'orange', 'blue', 'orange',  '#CC2200', 'green']
markers = ['x', '+', '*', 'x', '+', 'x', '+']
linestyles = ['-.', '-', '-.', '-', '--', ':', '-.', '-']

subset = (0, 1, 3, 4)

meths = [meths[i] for i in subset]
nm = [nm[i] for i in subset]
col = [col[i] for i in subset]
is_weighted = [is_weighted[i] for i in subset]

n_alpha = len(alphas)
n_meths = len(meths)

all_nmse = np.zeros((n_alpha, n_meths))

ii = 0
for log in log_loc:
    print(log)
    config = load_config(log + '/config')

    alpha_weight = config['optimizer']['alpha_weight']
    alpha = config['data']['sigma2'] / config['data']['sigma1']
    meth = config['data']['method']

    alpha = alphas.index(alpha) if alpha in alphas else 'none'

    meth_idx = 'none'
    for ii in range(len(meths)):
        if meths[ii] == meth and is_weighted[ii] in (alpha_weight, None):
            meth_idx = ii

    if os.path.exists(log + '/results.npz') and 'none' not in [alpha, meth_idx]:
        res = np.load(log + '/results.npz')
        if all_nmse[alpha, meth_idx] == 0:
            all_nmse[alpha, meth_idx] = np.mean(res[loss_type])
        else:
            print('double attempted at ({}, {}), file {}'.format(alpha, meth_idx, log))

    ii += 1

all_nmse[all_nmse == 0] = None
print(all_nmse)
sns.set_theme()

baseline_res = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984/results.npz'
baseline_nmse = np.mean(np.load(baseline_res)[loss_type])

fig = plt.figure(figsize=(5, 3))
#plt.subplot(121)

plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

all_nmse = 10 * np.log10(all_nmse)
baseline_nmse = 10 * np.log10(baseline_nmse)

for ii in [0, 1]:
    # prc_worse = 100 * all_nmse[0, :, ii] / all_nmse[0, :, 0] - 100
    prc_worse = 100 * all_nmse[:, ii] / baseline_nmse - 100
    prc_worse = all_nmse[:, ii] - baseline_nmse
    print(prc_worse)
    ax.plot(alphas, prc_worse, col[ii], marker=markers[ii],  linestyle=linestyles[ii])

ax.set_ylabel('$L - L_{Benchmark}$',  fontsize=14)
# ax.set_xlim([0.015, 0.085])
#ax.set_ylim([-5, 100])
ax.set_xlabel(r'Noise ratio $\alpha$',  fontsize=14)
# ax.set_title('$R_\Omega=8, \sigma_n = 0.06$')
plt.legend(nm, ncol=1, fontsize=13, loc='upper right')


# plt.subplot(122)
#
# for ii in [2, 3]:
#     # prc_worse = 100 * all_nmse[0, :, ii] / all_nmse[0, :, 0] - 100
#     prc_worse = 100 * all_nmse[:, ii] / baseline_nmse - 100
#     prc_worse = all_nmse[:, ii] -  baseline_nmse
#     print(prc_worse)
#     ax.plot(alphas, prc_worse, col[ii], marker=markers[ii],  linestyle=linestyles[ii])
#
# ax.set_ylabel('$L - L_{Benchmark}$',  fontsize=14)
# # ax.set_xlim([0.015, 0.085])
# #ax.set_ylim([-5, 100])
# ax.set_xlabel(r'Noise ratio $\alpha$',  fontsize=14)
# # ax.set_title('$R_\Omega=8, \sigma_n = 0.06$')
# plt.legend(nm, ncol=1, fontsize=13, loc='upper right')
#
fig.tight_layout()


plt.show()


