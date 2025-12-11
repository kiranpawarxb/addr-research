"""Entry point for running the GPU-CPU Hybrid Address Processing System.

This allows the hybrid processing system to be run with: python -m src.hybrid_main
or directly as: python src/hybrid_main.py
"""

import sys
from src.hybrid_cli import main

if __name__ == '__main__':
    sys.exit(main())