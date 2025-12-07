#!/usr/bin/env python3
"""
plot_raw.py
Scatter from raw_data.npz (freq_Hz, slowness_skm)
Run from folder containing raw_data.npz
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path



fp = Path.cwd() / 'raw_data.npz'
if not fp.exists():
    print('raw_data.npz not found')

dat = np.load(fp)
freq_hz = dat['freq_Hz']        # Hz
slow = dat['slowness_μsm']      # s/km

plt.figure(figsize=(7, 5))
plt.scatter(freq_hz / 1e3, 1/slow/freq_hz, s=8, c='k', marker='o')
plt.xlabel('Frequency, kHz')
plt.ylabel('Slowness, μs/m')
plt.title('Raw dispersion – all modes (from raw_data.npz)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

