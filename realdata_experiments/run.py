#!/usr/bin/env python
"""Usage: python run.py <run> <seed> <Ridge|MLP> [options]. See submit_all.sh for the settings used in the paper."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_common import main

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2:])
