"""Compatibility launcher for ``python main.py ...`` after package installation."""

from __future__ import annotations

import sys

from metro_navigation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["run", *sys.argv[1:]]))
