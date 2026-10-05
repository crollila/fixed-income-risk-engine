"""Pinned run configuration. Every number in the README derives from these inputs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
TEMPLATES_DIR = ROOT / "templates"

SEED = 7

# Valuation date: the last date of the checked-in FRED cache when this run was pinned.
VALUATION_DATE = date(2026, 10, 1)
