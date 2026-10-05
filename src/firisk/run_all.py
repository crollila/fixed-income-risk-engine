"""Reproduce every result, figure and document: ``python -m firisk.run_all``."""

from __future__ import annotations

import time

from firisk.config import FIGURES_DIR, RESULTS_DIR
from firisk.engine import run_analysis
from firisk.plots import make_all
from firisk.report import build_tokens, write_documents, write_results


def main() -> None:
    t0 = time.perf_counter()
    a = run_analysis()
    write_results(a, RESULTS_DIR)
    figs = make_all(a, FIGURES_DIR)
    tokens = build_tokens(a)
    docs = write_documents(tokens)
    v = a.validation
    print(f"valuation date      {a.market.valuation_date}  (curve: {a.history.source})")
    print("SIMULATED portfolio (not real AUM)")
    print(f"  market value      {tokens['p_nav']}")
    print(f"  eff. duration     {tokens['p_dur']}  vs benchmark {tokens['b_dur']}")
    print(f"  DV01 / CS01       {tokens['p_dv01']} / {tokens['p_cs01']}")
    print(f"  largest KR bucket {tokens['p_kr_top']} {tokens['p_kr_top_v']}")
    print(f"  worst stress      {tokens['worst_sc']}: {tokens['worst_pnl']}")
    print(f"  PCA PC1/2/3       {tokens['ev1']} / {tokens['ev2']} / {tokens['ev3']}")
    print(f"  rebalance         duration {tokens['p_dur']} -> {tokens['r_dur']}, turnover {tokens['turnover']}, cost {tokens['cost']}")
    print(f"checks: bootstrap err {v['bootstrap_max_abs_reprice_error']:.1e}, KR-vs-parallel {v['kr_sum_vs_parallel_rel_diff']:.2e}, "
          f"constraints pass={v['all_constraints_pass_after']}")
    print(f"wrote {len(figs)} figures, {len(docs)} documents, results/ in {time.perf_counter() - t0:.1f}s")
    if not v["all_constraints_pass_after"]:
        raise SystemExit("rebalance violated a constraint")


if __name__ == "__main__":
    main()
