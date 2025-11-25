import matplotlib.pyplot as plt
import numpy as np

def plot_slowness(results, model):
    """
    results = list of (f_kHz, omega, v) for each freq
    """
    fig, ax = plt.subplots()
    for f, w, v in results:
        slow = 1e6 / (w / (2*np.pi))  # us/ft
        ax.plot(f, slow, 'ro')
        ax.set_xlabel('f (kHz)')
        ax.set_ylabel('Slowness (μs/ft)')
        ax.set_title('Dispersion curves')
        plt.show()