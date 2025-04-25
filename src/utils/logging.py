from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(log_dir: str = 'logs') -> None:
    """Configure logging.

    Args:
        log_dir: Directory to store logs
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/training.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )
