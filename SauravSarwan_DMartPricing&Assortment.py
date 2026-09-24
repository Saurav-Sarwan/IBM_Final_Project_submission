"""
DMart Product Catalogue Analysis — Main Entry Point
=====================================================
Run this file to execute the full backend analysis pipeline
and generate all JSON output files consumed by the dashboard.

Usage:
    python "SauravSarwan_DMartPricing&Assortment.py"

After execution, open frontend/dashboard.html in any browser.
"""

import sys
import os

# Ensure the backend module is importable regardless of working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from analysis import run_analysis  # noqa: E402

if __name__ == "__main__":
    run_analysis()
