#!/usr/bin/env python3
"""Launcher. Equivalent to ``python -m vidtriage``."""

import sys

from vidtriage.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
