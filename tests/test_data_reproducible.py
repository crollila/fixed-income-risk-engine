import json
from datetime import date

import numpy as np
import pytest

from firisk.config import FIGURES_DIR, RESULTS_DIR, ROOT, VALUATION_DATE
from firisk.data import CurveHistory, load_history, synthetic_history
from firisk.plots import ALL_FIGURES
from firisk.report import DOCS, render, render_documents


def test_cache_is_present_and_complete_on_valuation_date():
    h = load_history()
    assert h.source == "FRED cache"
    y = h.par_yields_on(VALUATION_DATE)
    assert y.shape == (11,) and np.all((y > 0) & (y < 0.2))
    with pytest.raises(KeyError):
        h.par_yields_on(date(2026, 10, 4))  # a Sunday


def test_synthetic_fallback_is_deterministic_and_labeled(tmp_path):
    a, b = synthetic_history(), synthetic_history()
    assert a.equals(b)
    h = load_history(tmp_path / "missing.csv")
    assert h.source == "synthetic"
    assert isinstance(h, CurveHistory) and len(h.frame) == len(a)


def test_documents_rerender_exactly_from_saved_results():
    tokens = json.loads((RESULTS_DIR / "report_tokens.json").read_text(encoding="utf-8"))
    for name, text in render_documents(tokens).items():
        assert (ROOT / name).read_text(encoding="utf-8") == text, f"{name} out of date: run python -m firisk.run_all"


def test_tokens_match_numeric_results():
    tokens = json.loads((RESULTS_DIR / "report_tokens.json").read_text(encoding="utf-8"))
    s = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert tokens["p_dur"] == f"{s['portfolio']['duration']:.2f}"
    assert tokens["p_dv01"] == f"${s['portfolio']['dv01']:,.0f}"
    assert tokens["p_cs01"] == f"${s['portfolio']['cs01']:,.0f}"
    assert tokens["r_dur"] == f"{s['rebalanced']['duration']:.2f}"
    assert tokens["ev1"] == f"{s['pca']['explained'][0] * 100:.1f}%"
    assert s["validation"]["all_constraints_pass_after"] is True


def test_readme_is_explicit_about_simulation_and_has_figures():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Simulated portfolio, not real money" in readme
    assert "not client or real AUM" in readme
    pngs = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    assert len(pngs) == len(ALL_FIGURES)
    for name in pngs:
        assert f"figures/{name}" in readme
    assert set(DOCS) == {"README.md", "ANALYSIS.md", "RESUME_BULLETS.md"}


def test_render_rejects_unknown_tokens():
    with pytest.raises(KeyError):
        render("value {{nope}}", {})
