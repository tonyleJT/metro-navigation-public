"""Compatibility launcher for ``python benchmark_components.py ...`` after installation."""

from __future__ import annotations

import sys

from metro_navigation.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["benchmark", *sys.argv[1:]]))
