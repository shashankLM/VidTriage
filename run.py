#!/usr/bin/env python3
"""Launcher. Equivalent to ``python -m label_kit``."""

import sys

from label_kit.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
