"""
utils.py
========
Debug output and utilities (root level for all modules)
"""

import os
import sys
import time

# ANSI colors
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

DEBUG_LEVELS = {0: "ERROR", 1: "INFO", 2: "DEBUG", 3: "TRACE"}

def debug_print(message: str, level: int = 1, color=None):
    """Print debug message with color and level"""
    global_level = int(os.environ.get("SAFE_DEBUG_LEVEL", "1"))

    if level <= global_level:
        timestamp = time.strftime("%H:%M:%S")
        level_name = DEBUG_LEVELS.get(level, "UNKNOWN")

        if color is None:
            if level == 0:
                color = Colors.RED + Colors.BOLD
            elif level == 1:
                color = Colors.WHITE
            elif level == 2:
                color = Colors.CYAN
            else:
                color = Colors.GRAY

        prefix = f"[{timestamp}] [{level_name}]"
        print(f"{color}{prefix} {message}{Colors.ENDC}", flush=True)


class timer:
    """Context manager for timing operations"""
    def __init__(self, operation_name):
        self.operation_name = operation_name

    def __enter__(self):
        self.start_time = time.time()
        debug_print(f"Starting {self.operation_name}...", level=2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        debug_print(f"Completed {self.operation_name} in {elapsed:.3f}s", level=2)