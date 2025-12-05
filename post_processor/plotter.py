import matplotlib.pyplot as plt
import numpy as np


def plot_slowness(out, model):
    """Plot slowness dispersion curves with better mode filtering."""
    try:
        # Extract data from out
        freqs = np.array([item[0] for item in out])
        eigenvalues_list = [item[1] for item in out]

        num_freqs = len(freqs)
        num_modes = len(eigenvalues_list[0])

        # Calculate slowness and group by modes
        all_modes_data = []

        # For each frequency, organize eigenvalues by mode
        for mode_idx in range(num_modes):
            mode_slowness = []
            mode_freqs = []

            for i, (f, w_arr, _) in enumerate(out):
                if mode_idx < len(w_arr):
                    w = w_arr[mode_idx]
                    omega = 2 * np.pi * f * 1e3

                    # Calculate wave number (assuming w = k^2)
                    k = np.sqrt(np.real(w))

                    # if abs(k) > 1e-10 and abs(omega) > 1e-10:
                    phase_velocity = omega / k
                    slowness = 1.0 / phase_velocity
                    #
                    #     # Filter physically meaningful values
                    #     if 0 < slowness < 1e-3:  # reasonable slowness range for ultrasonic waves
                    mode_slowness.append(slowness)
                    mode_freqs.append(f)

            if len(mode_slowness) > 3:  # Only plot modes with enough data points
                all_modes_data.append((mode_freqs, mode_slowness))

        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, (mode_freqs, mode_slowness) in enumerate(all_modes_data):
            if len(mode_freqs) > 0:
                ax.plot(mode_freqs, mode_slowness*1e3, '.-', markersize=2, linewidth=1,
                        label=f'Mode {i + 1}')

        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel(r'Slowness ($\mu$s/m)')
        ax.set_title('Dispersion Curves')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        # Add model info
        model_info = f"Layers: {len(model['Model']['DomainType'])}\n"
        model_info += f"Total modes: {len(all_modes_data)}\n"
        model_info += f"Frequency: {freqs[0]:.1f}-{freqs[-1]:.1f} kHz"

        ax.text(0.02, 0.98, model_info, transform=ax.transAxes, fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))

        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error in plot_slowness: {e}")
        import traceback
        traceback.print_exc()