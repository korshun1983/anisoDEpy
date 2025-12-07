# routines/io_utils.py
"""
===============================================================================
I/O Utilities for SAFE Project
Handles file operations, MATLAB compatibility, and directory management
===============================================================================
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
from scipy.io import savemat

logger = logging.getLogger(__name__)


def cleanup_output_dir(output_dir: Path) -> None:
    """
    Removes all files from the output directory before new computation.
    Replicates MATLAB's delete('output/*.*') behavior.

    Args:
        output_dir: Path to output directory to clean
    """
    if not output_dir.exists():
        logger.debug(f"Output directory {output_dir} does not exist, creating it")
        output_dir.mkdir(parents=True, exist_ok=True)
        return

    # Remove all files
    removed_count = 0
    for file in output_dir.glob("*.*"):
        try:
            if file.is_file():
                file.unlink()
                removed_count += 1
        except Exception as e:
            logger.warning(f"Could not remove {file.name}: {e}")

    # Remove subdirectories
    for subdir in output_dir.iterdir():
        if subdir.is_dir():
            try:
                shutil.rmtree(subdir)
                removed_count += 1
            except Exception as e:
                logger.warning(f"Could not remove subdirectory {subdir.name}: {e}")

    if removed_count > 0:
        logger.debug(f"Cleaned {removed_count} items from {output_dir}")


def save_matlab_struct(file_path: Path, **kwargs: Dict[str, Any]) -> None:
    """
    Saves multiple variables to a MATLAB .mat file as a structure.

    Args:
        file_path: Output file path (should end with .mat)
        **kwargs: Keyword arguments become variables in the .mat file
    """
    try:
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save variables
        savemat(str(file_path), kwargs)
        logger.debug(f"Saved MATLAB struct to {file_path.name}")

    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")
        raise


def ensure_directory(dir_path: Path) -> Path:
    """
    Ensures a directory exists, creating it if necessary.

    Args:
        dir_path: Directory path to ensure

    Returns:
        Path to the directory (for chaining)
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def copy_if_exists(src: Path, dst: Path) -> bool:
    """
    Copies a file if it exists, logs warning if not.

    Args:
        src: Source file path
        dst: Destination file path

    Returns:
        True if file was copied, False if source doesn't exist
    """
    if src.exists() and src.is_file():
        try:
            shutil.copy2(src, dst)
            logger.debug(f"Copied {src.name} to {dst.parent}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy {src}: {e}")
            return False
    else:
        logger.warning(f"Source file not found: {src}")
        return False


def move_results_to_model_dir(output_dir: Path, model_dir: Path, pattern: str = "*.mat") -> int:
    """
    Moves computed results from output directory to model directory.

    Args:
        output_dir: Source directory (typically ./output)
        model_dir: Destination directory (typically ./models/Bakken-B)
        pattern: File glob pattern to match

    Returns:
        Number of files moved
    """
    if not output_dir.exists():
        logger.warning(f"Output directory {output_dir} does not exist")
        return 0

    model_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    for file in output_dir.glob(pattern):
        try:
            shutil.move(str(file), str(model_dir / file.name))
            moved += 1
        except Exception as e:
            logger.error(f"Failed to move {file.name}: {e}")

    if moved > 0:
        logger.info(f"Moved {moved} files to {model_dir}")

    return moved