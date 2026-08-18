"""Entry point for ``python -m vrp``."""

from __future__ import annotations

import sys

from vrp.cli import main

if __name__ == "__main__":
    sys.exit(main())
