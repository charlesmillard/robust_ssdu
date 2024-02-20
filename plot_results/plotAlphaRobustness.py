import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from configs.config_loader import *
import matplotlib.patches as mpatches

import matplotlib.ticker as ticker

rt = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/'
roots = ['alpha_robustness', 'alpha_robustness_n2n_ssdu', 'alpha_robustness_noise2recon',
         'whole_experiment/8x/0.06/', 'weighted_experiment/8x/0.06/',
         'whole_experiment_alph1/8x/0.06', 'whole_experiment_alph1_weighted/8x/0.06']
         # '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/weighted_experiment/8x/0.06/noise2recon.yaml',
         # '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/whole_experiment/8x/0.06/noise2recon.yaml']

roots = ['new_format/r2r_ssdu_rob', 'new_format/n2f_25_patch.yaml',
    'alpha_robustness_n2n_ssdu_weighted', 'alpha_robustness_n2n_ssdu_weighted/0.75.yaml/38770026_5',
         'whole_experiment_alph1_weighted_n2n_ssdu/8x/0.06/',
         'alpha_robustness_weighted_only', 'alpha_robustness', 'alpha_robustness_n2n_ssdu',
         'whole_experiment/8x/0.06/', 'whole_experiment_alph1/8x/0.06/',
         'weighted_experiment/8x/0.06/', 'old_whole_experiment_alph1_weighted/8x/0.06/']

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
meths = ['noisier2full', 'noisier2full', 'noise2recon', 'robust_ssdu',  'robust_ssdu', "r2r_ssdu"]
is_weighted = [False, True, None, False,  True, None]

# meths = ['n2n', 'n2n', 'n2n_ssdu',  'n2n_ssdu']
# is_weighted = [False, True, False,  True]

nm = ['Noisier2Full', 'Weighted Noisier2Full', 'Noise2Recon',
      'Robust SSDU', 'Weighted Robust SSDU', "Weighted R2R SSDU"]


col = ['black',  'black', 'red', 'green', 'orange', 'blue', 'green', 'red']
col = ['#CC2200', 'green', 'blue', 'black', 'orange', 'blue', 'orange',  '#CC2200', 'green']
markers = ['x', '+', '*', 'o', '+', 'x', '+']
linestyles = ['-.', '-', '-.', '-', '--', ':', '-.', '-']

subset = (0, 1, 3, 4, 5)
# subset = (3, 4, 5)

meths = [meths[i] for i in subset]
nm = [nm[i] for i in subset]
col = [col[i] for i in subset]
is_weighted = [is_weighted[i] for i in subset]

n_alpha = len(alphas)
n_meths = len(meths)

all_nmse = np.zeros((n_alpha, n_meths))

ii = 0
for log in log_loc:
    # print(log)
    config = load_config(log + '/config')
    config = reformat_config(config)
    # config = set_missing_config_entries(config)

    alpha_weight = config['optimizer']['alpha_weight']
    alpha = config['noise']['alpha']
    meth = config['optimizer']['method']

    alpha = alphas.index(alpha) if alpha in alphas else 'none'

    meth_idx = 'none'
    for ii in range(len(meths)):
        if meths[ii] == meth and is_weighted[ii] in (alpha_weight, None):
            meth_idx = ii

    if os.path.exists(log + '/results.npz') and 'none' not in [alpha, meth_idx]:
        res = np.load(log + '/results.npz')
        if all_nmse[alpha, meth_idx] == 0:
            all_nmse[alpha, meth_idx] = np.mean(res[loss_type][~np.isnan(res[loss_type])])

            print('alpha: {}; method: {}; weighted: {}; log: {}'.format(alphas[alpha], meths[meth_idx], alpha_weight, log))

        else:
            print('double attempted at alpha: {}; method: {}; weighted: {}; log: {}'.format(alphas[alpha], meths[meth_idx], alpha_weight, log))

    ii += 1

all_nmse[all_nmse == 0] = None
print(all_nmse)
sns.set_theme()

baseline_res = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/whole_experiment/8x/0.06/full.yaml/16422984/results.npz'
# baseline_res = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/new_format/full_trunc.yaml/47550499_1/results.npz'
# baseline_res = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/new_format/full_random_omega.yaml/47666062_1/results.npz'
print('baseline is {}'.format(baseline_res))
res = np.load(baseline_res)
baseline_nmse = np.mean(res[loss_type][~np.isnan(res[loss_type])])
print(baseline_nmse)

fig = plt.figure(figsize=(5, 3))
# plt.subplot(121)

plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

ax = plt.axes()
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.25))

all_nmse =  10 * np.log10(all_nmse)
baseline_nmse =  10 * np.log10(baseline_nmse)

for ii in range(n_meths):
    # prc_worse = 100 * all_nmse[0, :, ii] / all_nmse[0, :, 0] - 100
    # prc_worse = 100 * all_nmse[:, ii] / baseline_nmse - 100
    prc_worse = all_nmse[:, ii] -  baseline_nmse
    #prc_worse = -prc_worse
    print(prc_worse)
    ax.plot(alphas, prc_worse, col[ii], marker=markers[ii],  linestyle=linestyles[ii])

#ax.set_ylabel('$L_{Benchmark} - L$',  fontsize=14)
ax.set_ylabel('$L - L_{Benchmark}$',  fontsize=14)
# ax.set_xlim([0.015, 0.085])
#ax.set_ylim([-5, 100])
ax.set_xlabel(r'Noise ratio $\alpha$',  fontsize=14)
# ax.set_title('$R_\Omega=8, \sigma_n = 0.06$')
plt.legend(nm, ncol=1, fontsize=13, loc='upper right')
fig.tight_layout()
plt.show()


