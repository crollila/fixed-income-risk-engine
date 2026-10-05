"""Compare two results directories numerically (used by CI after a fresh run).

Byte equality is too strict across platforms (BLAS / libm differ in the last
bits), so numbers are compared with a tight tolerance and everything else
exactly::

    python -m firisk.compare committed_results/ results/
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

RTOL, ATOL = 1e-8, 1e-6


def _close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    return a == b


def compare_dirs(expected: Path, actual: Path) -> list[str]:
    problems = []
    exp_files = sorted(p.name for p in expected.iterdir() if p.is_file())
    act_files = sorted(p.name for p in actual.iterdir() if p.is_file())
    if exp_files != act_files:
        problems.append(f"file sets differ: {sorted(set(exp_files) ^ set(act_files))}")
    for name in sorted(set(exp_files) & set(act_files)):
        e, a = expected / name, actual / name
        if name.endswith(".csv"):
            de, da = pd.read_csv(e), pd.read_csv(a)
            if list(de.columns) != list(da.columns) or de.shape != da.shape:
                problems.append(f"{name}: shape/columns differ")
                continue
            for c in de.columns:
                numeric = all(pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c]) for d in (de, da))
                if numeric:
                    bad = ~((de[c] - da[c]).abs() <= ATOL + RTOL * da[c].abs()) & ~(de[c].isna() & da[c].isna())
                    if bad.any():
                        problems.append(f"{name}:{c} differs in {int(bad.sum())} rows")
                elif not de[c].equals(da[c]):
                    problems.append(f"{name}:{c} text differs")
        elif name.endswith(".json"):
            if not _close(json.loads(e.read_text(encoding="utf-8")), json.loads(a.read_text(encoding="utf-8"))):
                problems.append(f"{name}: differs")
        elif e.read_bytes() != a.read_bytes():
            problems.append(f"{name}: bytes differ")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description="numerically compare two results directories")
    ap.add_argument("expected", type=Path)
    ap.add_argument("actual", type=Path)
    args = ap.parse_args()
    problems = compare_dirs(args.expected, args.actual)
    for p in problems:
        print("MISMATCH", p)
    if problems:
        sys.exit(1)
    print(f"results match ({args.expected} vs {args.actual}, rtol={RTOL}, atol={ATOL})")


if __name__ == "__main__":
    main()
