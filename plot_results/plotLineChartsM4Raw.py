import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from utils.preparations import load_config
import matplotlib.patches as mpatches

import seaborn as sns

root = '/home/xsd618/noisier2noise_kspace_denoising/logs/cuda/m4raw/'

# all_logs = ['full.yaml/32966674', 'ssdu.yaml/32917761', 'other_arch/n2n_ssdu_weighted.yaml/33247802']

all_logs = ['full_4x.yaml/33075783', 'ssdu_4x.yaml/33247656', 'other_arch/n2n_ssdu_weighted_4x.yaml/33247799']


av = np.mean

all_nmse = []
all_ssim = []
all_time = []
for log in all_logs:
    if os.path.exists(root + log + '/results.npz'):
        res = np.load(root + log + '/results.npz')
        non_nans = ~np.isnan(res['loss_rss']) * ~np.isnan(res['loss_rss_bm3d']) * ~np.isnan(res['all_ssim'])
        mean_loss = av(res['loss_rss'][non_nans])
        mean_time = av(res['time_all'][non_nans])
        mean_ssim = av(res['all_ssim'][non_nans])
        all_nmse.append(mean_loss)
        all_time.append(mean_time)
        all_ssim.append(mean_ssim)
        if log[:4] == 'ssdu':
            mean_loss = av(res['loss_rss_bm3d'][non_nans])
            mean_time = av(res['time_all_bm3d'][non_nans])
            mean_ssim = av(res['all_ssim_bm3d'][non_nans])
            all_nmse.append(mean_loss)
            all_time.append(mean_time)
            all_ssim.append(mean_ssim)

names = ['Supervised to noisy', 'SSDU', 'SSDU with BM3D', 'Robust SSDU']
col = ['black', 'orange', 'green', 'blue']

print(all_ssim, all_nmse, all_time)
sns.set_theme()

fig, ax = plt.subplots()
ax.bar(np.arange(0, len(all_time)), all_time, color=col)
# plt.bar(np.arange(0, len(all_time)), all_time)
plt.xticks(np.arange(0, len(all_time)), names)
plt.ylabel('Average reconstruction time (s)')

fig, ax = plt.subplots()
ax.bar(np.arange(0, len(all_nmse)), all_nmse, color=col, width=0.6)
plt.xticks(np.arange(0, len(all_nmse)), names)
plt.ylabel('Reconstruction NMSE')

fig, ax = plt.subplots()
ax.bar(np.arange(0, len(all_ssim)), all_ssim, color=col)
plt.xticks(np.arange(0, len(all_ssim)), names)
plt.ylabel('Reconstruction SSIM')

plt.show()


