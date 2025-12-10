#!/usr/bin/env python3
"""
===============================================================================
Raw Spectrum Plotter - Python Implementation
Replicates plot_raw_spectrum.m functionality
Prompts user via GUI for directory selection (or uses command-line argument)
===============================================================================
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def select_results_directory(initial_dir: Optional[Path] = None) -> Path:
    """
    Opens GUI dialog for user to select directory containing Results-*.mat files.

    Args:
        initial_dir: Starting directory for file dialog. Defaults to current working directory.

    Returns:
        Path to selected directory

    Raises:
        SystemExit: If no directory is selected
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)

    if initial_dir is None:
        initial_dir = Path.cwd()

    selected_dir = filedialog.askdirectory(
        title="Select Directory with Results-*.mat Files",
        initialdir=str(initial_dir)
    )

    if not selected_dir:
        logger.error("No directory selected. Exiting.")
        sys.exit(1)

    selected_path = Path(selected_dir)
    logger.info(f"Selected directory: {selected_path}")
    return selected_path


def find_results_files(search_dir: Path) -> List[Path]:
    """
    Find all Results-*.mat files in the specified directory.

    Args:
        search_dir: Directory to search for Results files

    Returns:
        List of Paths to Results files (sorted by numerical value)
    """
    files = list(search_dir.glob("Results-*.mat"))
    if not files:
        raise FileNotFoundError(f"No Results-*.mat files found in {search_dir}")

    # Sort by frequency value (extract number from filename)
    files.sort(key=lambda f: float(f.stem.split('-')[1]))

    logger.info(f"Found {len(files)} Results files")
    return files


def extract_frequency_and_slowness(results_file: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract frequency (Hz) and slowness (s/m) from a Results-*.mat file.

    Args:
        results_file: Path to Results-*.mat file

    Returns:
        Tuple of (frequencies, slownesses, base_frequency)
    """
    try:
        data = loadmat(results_file, struct_as_record=False, squeeze_me=True)

        if 'Results' not in data:
            raise KeyError(f"'Results' not found in {results_file.name}")

        Results = data['Results']

        # Extract required fields
        if isinstance(Results, dict):
            # Modern MATLAB structure format
            omega_val = float(Results.get('omega_val', 0.0))
            eig_vals = Results.get('REig_vals', np.array([]))
        else:
            # Old format or cell array
            logger.warning(f"Unexpected Results format in {results_file.name}, skipping")
            return np.array([]), np.array([]), 0.0

        if eig_vals.size == 0:
            logger.warning(f"No eigenvalues in {results_file.name}")
            return np.array([]), np.array([]), 0.0

        # Extract diagonal eigenvalues (k - wavenumbers)
        if eig_vals.ndim == 2 and eig_vals.shape[0] == eig_vals.shape[1]:
            k_values = np.diag(eig_vals)  # Extract diagonal (MATLAB format)
        else:
            k_values = eig_vals.flatten()

        # Compute frequency in Hz: f = ω / (2π)
        frequency_hz = omega_val / (2 * np.pi)

        # Compute slowness: s = k / ω (units: s/m)
        slownesses = k_values / omega_val if omega_val != 0 else np.zeros_like(k_values)

        # Filter out invalid values (NaN, Inf)
        valid_mask = np.isfinite(slownesses)
        k_values = k_values[valid_mask]
        slownesses = slownesses[valid_mask]

        n_modes = len(k_values)
        frequencies = np.full(n_modes, frequency_hz)

        logger.debug(
            f"File {results_file.name}: f={frequency_hz:.2f} Hz, "
            f"{n_modes} modes, slowness range=[{np.min(slownesses):.2e}, {np.max(slownesses):.2e}] s/m"
        )

        return frequencies, slownesses, frequency_hz

    except Exception as e:
        logger.error(f"Error loading {results_file.name}: {e}")
        return np.array([]), np.array([]), 0.0


def load_all_spectra(results_files: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and concatenate data from all Results files.

    Args:
        results_files: List of Results file paths

    Returns:
        Tuple of (all_frequencies, all_slownesses)
    """
    all_freqs = []
    all_slows = []

    for results_file in results_files:
        try:
            freqs, slows, base_freq = extract_frequency_and_slowness(results_file)

            if len(freqs) > 0:
                all_freqs.append(freqs)
                all_slows.append(slows)

        except Exception as e:
            logger.warning(f"Skipping {results_file.name}: {e}")
            continue

    if not all_freqs:
        raise ValueError("No valid data found in any Results file")

    # Concatenate all arrays
    frequencies = np.concatenate(all_freqs)
    slownesses = np.concatenate(all_slows)

    return frequencies, slownesses


def plot_raw_spectrum(frequencies: np.ndarray, slownesses: np.ndarray,
                      output_dir: Optional[Path] = None,
                      units: str = 'us/m') -> plt.Figure:
    """
    Create scatter plot of raw dispersion spectrum.

    Args:
        frequencies: Frequency values in Hz
        slownesses: Slowness values in s/m
        output_dir: Optional directory to save plot
        units: Display units ('us/m', 'us/ft', 's/m', 's/ft')

    Returns:
        Matplotlib Figure object
    """
    # Convert units for display
    if units == 'us/m':
        scale = 1e6  # s/m to µs/m
        ylabel = 'Slowness, µs/m'
    elif units == 'us/ft':
        scale = 1e6 * 0.3048  # s/m to µs/ft
        ylabel = 'Slowness, µs/ft'
    elif units == 's/ft':
        scale = 0.3048  # s/m to s/ft
        ylabel = 'Slowness, s/ft'
    else:  # 's/m'
        scale = 1.0
        ylabel = 'Slowness, s/m'

    # Apply scaling
    slowness_display = slownesses * scale

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    scatter = ax.scatter(frequencies / 1e3, slowness_display,  # Convert Hz to kHz
                         s=12, c='black', marker='o', alpha=0.6,
                         edgecolors='none')

    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_box_aspect(0.7)  # Similar to MATLAB's box on

    ax.set_xlabel('Frequency, kHz')
    ax.set_ylabel(ylabel)
    ax.set_title('Raw Dispersion Spectrum – All Modes')

    # Set ticks
    freq_min, freq_max = frequencies.min() / 1e3, frequencies.max() / 1e3
    ax.set_xticks(np.arange(np.floor(freq_min), np.ceil(freq_max) + 0.25, 0.25))

    # Auto-scale y-axis
    if slowness_display.max() > slowness_display.min() * 10:
        ax.set_yscale('log')

    plt.tight_layout()

    # Save plot if output directory provided
    if output_dir:
        plot_file = output_dir / "raw_spectrum.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved plot: {plot_file}")

    return fig


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Plot raw dispersion spectrum from SAFE Results-*.mat files"
    )
    parser.add_argument(
        'directory',
        type=str,
        nargs='?',
        help="Directory containing Results-*.mat files (optional, will prompt if not provided)"
    )
    parser.add_argument(
        '--units',
        type=str,
        choices=['us/m', 'us/ft', 's/m', 's/ft'],
        default='us/m',
        help="Slowness units for display (default: us/m)"
    )
    parser.add_argument(
        '--output',
        type=str,
        help="Output directory to save plot (optional)"
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point with GUI fallback"""
    args = parse_arguments()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    # Determine search directory: CLI argument > GUI prompt
    if args.directory:
        search_dir = Path(args.directory).resolve()
        if not search_dir.exists():
            logger.error(f"Directory does not exist: {search_dir}")
            sys.exit(1)
    else:
        # Prompt user via GUI
        print("\n" + "=" * 70)
        print("Raw Spectrum Plotter - GUI Mode")
        print("=" * 70)
        search_dir = select_results_directory()

    # Resolve output directory
    output_dir = Path(args.output).resolve() if args.output else None

    try:
        logger.info(f"\nProcessing directory: {search_dir}")

        # Find and load files
        results_files = find_results_files(search_dir)
        frequencies, slownesses = load_all_spectra(results_files)

        logger.info(
            f"\nLoaded {len(frequencies)} modes from {len(results_files)} frequencies"
        )
        logger.info(
            f"Frequency range: {frequencies.min() / 1e3:.2f} - {frequencies.max() / 1e3:.2f} kHz"
        )
        logger.info(
            f"Slowness range: {slownesses.min():.2e} - {slownesses.max():.2e} s/m"
        )

        # Create plot
        fig = plot_raw_spectrum(frequencies, slownesses, output_dir, args.units)

        # Show plot (blocking)
        plt.show()

        logger.info("✓ Plot completed successfully")

    except Exception as e:
        logger.error(f"\n{'=' * 70}\nPLOTTER FAILED: {str(e)}\n{'=' * 70}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()