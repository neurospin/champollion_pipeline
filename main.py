#!/usr/bin/env python3
"""Champollion pipeline: ``python main.py --help`` lists the options.

Thin shim kept for ``python main.py``. The orchestrator and the command line live in
champollion_pipeline.cli, installed as the ``champollion-pipeline`` console script.
"""

import sys

from champollion_pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())
