import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from configs.config_loader import *
import matplotlib.patches as mpatches

root_loc = '/home/xsd618/noisier2noise_kspace_denoising/saved/logs/cuda/'
loss_type = 'loss'

roots = [# 'just_n2n_ssdu_alph0.5',
         # 'whole_experiment/8x/0.06/noise2recon.yaml',
         # 'whole_experiment_alph1',
        #'whole_experiment_alph1/4x/0.02/noise2recon.yaml',
        # 'whole_experiment_alph1/4x/0.04/noise2recon.yaml',
         'whole_experiment_alph1/4x/0.06/noise2recon.yaml',
        'whole_experiment_alph1/4x/0.08/noise2recon.yaml',
        # 'whole_experiment_alph1/8x/0.02/noise2recon.yaml',
        'whole_experiment_alph1/8x/0.04/noise2recon.yaml',
        #'whole_experiment_alph1/8x/0.06/noise2recon.yaml',
        'alpha_robustness_weighted_only',
        'whole_experiment_alph1/8x/0.08/noise2recon.yaml',
        'whole_experiment_alph1_weighted',
        'whole_experiment']

# roots = ['whole_experiment', 'weighted_experiment']

log_loc = []
for r in roots:
    for f in os.walk(root_loc + r):
        if 'state_dict' in f[2]:
            log_loc.append(f[0])


accels = [4, 8]
sigmas = [0.02, 0.04, 0.06, 0.08]
meths = ['full', 'noise2full', 'noisier2full', 'noisier2full', \
         'ssdu', 'noise2recon', 'robust_ssdu',  'robust_ssdu']
is_weighted = [None, None, False, True, None, None, False,  True]

nm = [ 'Baseline: Supervised to clean',
      'Noise2Full', 'Noisier2Full', \
      'Weighted Noisier2Full', 'Standard SSDU', 'Noise2Recon-SS',
      'Robust SSDU', 'Weighted Robust SSDU']
col = ['black',  'black', 'red', 'green', 'orange', 'blue', 'green', 'red']
col = ['', 'blue', '#CC2200', 'green', 'blue', 'orange',  '#CC2200', 'green']
markers = ['+', 'o', 'x', '+', 'o', '*', 'x', '+']
linestyles = [':', '--', '-.', '-', '--', ':', '-.', '-']

#subset = (0, 1, 2, 3, 4, 5, 6, 7) # everything
subset = (0, 1, 2, 3) # noisy, fully sampled training data
#subset = (0, 4, 5, 6, 7) # noisy, sub-sampled sampled training data

subset = (0, 1, 2, 3) # noisy, fully sampled training data weighted only
#subset = (0, 4, 5, 6, 7) # noisy, sub-sampled sampled training data weighted only
# subset = (0, 4, 5, 7)

meths = [meths[i] for i in subset]
nm = [nm[i] for i in subset[1:]]
col = [col[i] for i in subset]
markers = [markers[i] for i in subset]
linestyles = [linestyles[i] for i in subset]
is_weighted = [is_weighted[i] for i in subset]
print(meths)

n_accels = len(accels)
n_sigmas = len(sigmas)
n_meths = len(meths)

all_nmse = np.zeros((n_accels, n_sigmas, n_meths))

ii = 0
for log in log_loc:

    config = load_config(log + '/config')
    config = reformat_config(config)

    ac = config['mask']['us_fac']
    sig = config['noise']['sigma1']
    meth = config['optimizer']['method']
    k_weighted = config['optimizer']['K_weight']

    print(meth)

    ac_idx = accels.index(ac) if ac in accels else 'none'
    sig_idx = sigmas.index(sig) if sig in sigmas else 'none'

    meth_idx = 'none'
    for ii in range(len(meths)):
        if meths[ii] == meth and is_weighted[ii] in (k_weighted, None):
            meth_idx = ii

   #  meth_idx = meths.index(meth) if meth in meths else 'none'

    print(meth_idx)

    if os.path.exists(log + '/results.npz') and 'none' not in [ac_idx, sig_idx, meth_idx]:
        res = np.load(log + '/results.npz')
        mean_loss = np.mean(res[loss_type][~np.isnan(res[loss_type])])
        if all_nmse[ac_idx, sig_idx, meth_idx] == 0:# or mean_loss < all_nmse[ac_idx, sig_idx, meth_idx]:
            all_nmse[ac_idx, sig_idx, meth_idx] = mean_loss
            print('accel: {}; method: {}; weighted: {}; log: {}'.format(accels[ac_idx], meths[meth_idx], k_weighted, log))
        else:
            print('double attempted at ({}, {}, {}), file {}'.format(ac_idx, sig_idx, meth_idx, log))

    ii += 1

all_nmse[all_nmse == 0] = None
print(all_nmse)
sns.set_theme()


fig = plt.figure(figsize=(6, 3))
# plt.subplot(121)

plt.rcParams.update({'font.family': 'Linux Libertine Display O'})
plt.rc('axes', labelsize=20)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
ax = fig.add_subplot(121)

all_nmse = 10 * np.log10(all_nmse)

for ii in range(1, n_meths):
    prc_worse = 100 * all_nmse[0, :, ii] / all_nmse[0, :, 0] - 100
    prc_worse = all_nmse[0, :, ii] -  all_nmse[0, :, 0]
    # prc_worse = all_nmse[0, :, ii]
    # prc_worse = (all_nmse[0, :, ii]) - (all_nmse[0, 0, 0])
    print(prc_worse)
    ax.plot(sigmas, prc_worse, col[ii], marker=markers[ii],  linestyle=linestyles[ii])
ax.set_ylabel('$L - L_{Benchmark}$',  fontsize=14)
# ax.set_xlim([0.015, 0.085])
ax.set_ylim([-0.1, 3.3])
ax.set_xlabel('Measurement noise $\sigma_n$',  fontsize=14)
ax.set_title('$R_\Omega=4$')
plt.legend(nm, ncol=1, fontsize=10, loc='upper left')

ax2 = fig.add_subplot(122)
for ii in range(1, n_meths):
    prc_worse = 100 * all_nmse[1, :, ii] / all_nmse[1, :, 0] - 100
    prc_worse = all_nmse[1, :, ii] - all_nmse[1, :, 0]
    print(prc_worse)
    ax2.plot(sigmas, prc_worse, col[ii], marker=markers[ii],  linestyle=linestyles[ii])
ax2.set_ylabel('$L - L_{Benchmark}$',  fontsize=14)
# ax2.set_xlim([0.015, 0.085])
ax2.set_ylim([-0.1, 3.3])
ax2.set_xlabel('Measurement noise $\sigma_n$',  fontsize=14)
ax2.set_title('$R_\Omega=8$')
fig.tight_layout()


plt.show()


