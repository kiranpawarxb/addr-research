"""Entry point for running the Address Consolidation System as a module.

This allows the package to be run with: python -m src
"""

import sys
from src.cli import main

if __name__ == '__main__':
    sys.exit(main())
