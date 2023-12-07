import os
import shutil

roots = ['/home/xsd618/noisier2noise_kspace_denoising/logs/cuda']

all_logs = []
for r in roots:
    for f in os.walk(r):
        if 'state_dict' not in f[2] and "config" in f[2]:
            shutil.rmtree(f[0])
            all_logs.append(f[0])

print(all_logs)
