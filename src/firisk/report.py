"""Write machine-readable results and render the Markdown documents from them.

Every number in README.md / ANALYSIS.md / RESUME_BULLETS.md is a ``{{token}}``
in ``templates/*.md`` replaced from ``results/report_tokens.json``, which is
itself produced from the analysis in the same run. ``tests/test_reproducible``
re-renders the documents from the saved tokens and requires an exact match.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from firisk.config import RESULTS_DIR, ROOT, TEMPLATES_DIR
from firisk.data import TENOR_LABELS
from firisk.engine import PCA_TENORS, Analysis

DOCS = ["README.md", "ANALYSIS.md", "RESUME_BULLETS.md"]
TOKEN = re.compile(r"\{\{\s*([A-Za-z0-9_:]+)\s*\}\}")


def usd(x: float) -> str:
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def mm(x: float, nd: int = 2) -> str:
    return f"-${abs(x) / 1e6:,.{nd}f}MM" if x < 0 else f"${x / 1e6:,.{nd}f}MM"


def smm(x: float, nd: int = 2) -> str:
    return ("+" if x >= 0 else "") + mm(x, nd)


def pct(x: float, nd: int = 1) -> str:
    return f"{x * 100:.{nd}f}%"


def _bound(x: float, limit: float) -> str:
    return f"below {limit:.0e}" if abs(x) < limit else f"{x:.1e}"


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(cols))) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def _round_frame(df: pd.DataFrame, nd: int = 6) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].round(nd)
    return out


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items() if k != "problem"}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _book_summary(p) -> dict:
    iss, sec = p.group_weights("issuer"), p.group_weights("sector")
    return {"nav": p.nav, "cash": p.cash, "duration": p.duration, "dv01": p.dv01, "cs01": p.cs01,
            "spread_duration": p.spread_duration, "convexity": p.convexity, "ytm": p.ytm, "dts": p.dts(),
            "treasury_weight": p.treasury_weight, "kr_dv01": dict(zip(TENOR_LABELS, p.kr_dv01, strict=True)),
            "max_issuer": [iss.index[0], iss.iloc[0]], "max_sector": [sec.index[0], sec.iloc[0]],
            "issuer_weights": iss.to_dict(), "sector_weights": sec.to_dict(),
            "credit_by_rating": p.credit_exposure().to_dict(orient="index")}


def write_results(a: Analysis, out: Path = RESULTS_DIR) -> dict:
    out.mkdir(parents=True, exist_ok=True)

    def csv(df: pd.DataFrame, name: str, index: bool = False) -> None:
        _round_frame(df).to_csv(out / name, index=index, lineterminator="\n")

    csv(a.portfolio.positions, "holdings_before.csv")
    csv(a.rebalanced.positions, "holdings_after.csv")
    csv(a.risk.frame, "bond_analytics.csv")
    kr = pd.DataFrame({"tenor": TENOR_LABELS, "portfolio": a.portfolio.kr_dv01, "benchmark": a.benchmark.kr_dv01,
                       "rebalanced": a.rebalanced.kr_dv01})
    csv(kr, "key_rate_dv01.csv")
    csv(a.stress, "stress_before.csv")
    csv(a.stress_after, "stress_after.csv")
    csv(a.pca_scenarios, "stress_pca_factors.csv")
    csv(a.convergence, "approx_convergence.csv")
    c = a.market.curve
    curve = pd.DataFrame({"tenor": TENOR_LABELS, "years": a.market.tenors, "par_yield": a.market.par_yields,
                          "zero_cont": c.zero_rate(a.market.tenors), "zero_semi": c.zero_rate(a.market.tenors, "semi"),
                          "discount_factor": c.df(a.market.tenors)})
    csv(curve, "curve.csv")
    load = pd.DataFrame(a.pca.loadings[:, :3], columns=["PC1", "PC2", "PC3"])
    load.insert(0, "tenor", PCA_TENORS)
    csv(load, "pca_loadings.csv")
    csv(pd.DataFrame({"pc": [f"PC{i + 1}" for i in range(len(a.pca.explained))], "eigenvalue_bp2": a.pca.eigenvalues,
                      "explained": a.pca.explained}), "pca_explained.csv")
    csv(a.trades[a.trades["trade_face"] != 0], "rebalance_trades.csv")
    csv(pd.concat([a.constraints_before.assign(book="before"), a.constraints_after.assign(book="after")]),
        "rebalance_constraints.csv")
    csv(a.portfolio.ladder(), "ladder_before.csv", index=True)
    csv(a.rebalanced.ladder(), "ladder_after.csv", index=True)

    summary = {
        "valuation_date": a.market.valuation_date.isoformat(), "curve_source": a.history.source,
        "portfolio": _book_summary(a.portfolio), "benchmark": _book_summary(a.benchmark),
        "rebalanced": _book_summary(a.rebalanced), "rebalance": a.rebalance_terms,
        "pca": {"tenors": PCA_TENORS, "explained": a.pca.explained[:5], "n_obs": a.pca.n_obs, "start": a.pca.start,
                "end": a.pca.end, "labels": [a.pca.shape_label(j) for j in range(3)],
                "all_tenor_explained": a.pca_all_tenors.explained[:5],
                "all_tenor_pc3_1m_loading": a.pca_all_tenors.loadings[0, 2]},
        "factor_risk_before": a.factor_before, "factor_risk_after": a.factor_after,
        "hist_var_before": a.hist_var_before, "hist_var_after": a.hist_var_after,
        "validation": a.validation,
    }
    summary = _clean(summary)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def count_tests(tests_dir: Path = ROOT / "tests") -> int:
    """Number of test functions (parametrized cases count once)."""
    return sum(len(re.findall(r"^\s*def test_", f.read_text(encoding="utf-8"), re.M))
               for f in sorted(tests_dir.glob("test_*.py")))


def build_tokens(a: Analysis) -> dict[str, str]:
    p, b, r = a.portfolio, a.benchmark, a.rebalanced
    f = a.risk.frame
    t: dict[str, str] = {}
    t["n_tests"] = str(count_tests())
    t["val_date"] = a.market.valuation_date.strftime("%d %B %Y").lstrip("0")
    t["val_date_iso"] = a.market.valuation_date.isoformat()
    t["curve_source"] = a.history.source
    t["hist_start"] = str(a.history.frame.index[0].date())
    t["hist_end"] = str(a.history.frame.index[-1].date())
    t["hist_rows"] = f"{len(a.history.frame):,}"
    t["par10"] = pct(a.market.par_yields[TENOR_LABELS.index("10Y")], 2)
    t["par2"] = pct(a.market.par_yields[TENOR_LABELS.index("2Y")], 2)
    t["par30"] = pct(a.market.par_yields[TENOR_LABELS.index("30Y")], 2)
    t["n_universe"] = str(len(f))
    t["n_tsy_universe"] = str(int((f["kind"] == "TSY").sum()))
    t["n_corp_universe"] = str(int((f["kind"] == "CORP").sum()))
    held = p.positions
    t["n_held"] = str(len(held))
    t["n_tsy_held"] = str(int((held["kind"] == "TSY").sum()))
    t["n_corp_held"] = str(int((held["kind"] == "CORP").sum()))
    t["n_issuers"] = str(held.loc[held["kind"] == "CORP", "issuer"].nunique())
    t["n_sectors"] = str(held.loc[held["kind"] == "CORP", "sector"].nunique())

    for key, book in (("p", p), ("b", b), ("r", r)):
        t[f"{key}_nav"] = usd(book.nav)
        t[f"{key}_nav_mm"] = mm(book.nav, 2)
        t[f"{key}_dur"] = f"{book.duration:.2f}"
        t[f"{key}_dv01"] = usd(book.dv01)
        t[f"{key}_cs01"] = usd(book.cs01)
        t[f"{key}_sdur"] = f"{book.spread_duration:.2f}"
        t[f"{key}_conv"] = f"{book.convexity:.1f}"
        t[f"{key}_ytm"] = pct(book.ytm, 2)
        t[f"{key}_dts"] = f"{book.dts():,.0f}"
        t[f"{key}_tsy"] = pct(book.treasury_weight)
        iss, sec = book.group_weights("issuer"), book.group_weights("sector")
        t[f"{key}_max_issuer"] = iss.index[0]
        t[f"{key}_max_issuer_w"] = pct(iss.iloc[0], 2)
        t[f"{key}_max_sector"] = sec.index[0]
        t[f"{key}_max_sector_w"] = pct(sec.iloc[0], 2)
        k = int(np.argmax(book.kr_dv01))
        t[f"{key}_kr_top"] = TENOR_LABELS[k]
        t[f"{key}_kr_top_v"] = usd(book.kr_dv01[k])
        t[f"{key}_kr_top_share"] = pct(book.kr_dv01[k] / book.dv01)
        t[f"{key}_kr_long"] = pct(book.kr_dv01[TENOR_LABELS.index("20Y"):].sum() / book.dv01)
    t["dts_ratio"] = f"{p.dts() / b.dts():.1f}"
    t["active_dur"] = f"{p.duration - b.duration:+.2f}"
    t["r_active_dur"] = f"{r.duration - b.duration:+.2f}"
    t["dv01_per_mm"] = usd(p.dv01 / (p.nav / 1e6))
    t["kr_sum"] = usd(p.kr_dv01.sum())
    t["kr_recon"] = f"{a.validation['kr_sum_vs_parallel_rel_diff'] * 100:.5f}%"
    t["kr_recon_bond"] = f"{a.validation['max_bond_kr_sum_vs_parallel_rel_diff'] * 100:.4f}%"
    # machine-precision quantities are reported as bounds so the text is platform independent
    t["boot_err"] = _bound(a.validation["bootstrap_max_abs_reprice_error"], 1e-12)
    t["dv01_full"] = f"${a.validation['portfolio_dv01_full_reprice']:,.4f}"
    t["dv01_sum"] = f"${a.validation['portfolio_dv01_sum_of_positions']:,.4f}"
    t["dv01_sum_diff"] = _bound(abs(a.validation["portfolio_dv01_full_reprice"] - a.validation["portfolio_dv01_sum_of_positions"]), 1e-6)
    cr = p.credit_exposure()
    for rating in ("AA", "A", "BBB"):
        t[f"p_mv_{rating}"] = pct(cr.loc[rating, "weight"])
        t[f"p_cs01_{rating}"] = usd(cr.loc[rating, "cs01"])
    t["p_corp_w"] = pct(1 - p.treasury_weight)

    s = a.stress.set_index("scenario")
    for name, key in (("Parallel +100bp", "up100"), ("Parallel -100bp", "dn100"), ("Parallel +25bp", "up25"),
                      ("Parallel +50bp", "up50")):
        row = s.loc[name]
        t[f"{key}_full"] = smm(row["full_pnl"])
        t[f"{key}_first"] = smm(row["first_order_pnl"])
        t[f"{key}_second"] = smm(row["second_order_pnl"])
        t[f"{key}_err1"] = usd(row["first_order_error"])
        t[f"{key}_err2"] = usd(row["second_order_error"])
        t[f"{key}_err1_pct"] = pct(abs(row["first_order_error"] / row["full_pnl"]), 2)
        t[f"{key}_err2_pct"] = pct(abs(row["second_order_error"] / row["full_pnl"]), 2)
        t[f"{key}_pct_nav"] = pct(row["full_pnl_pct_nav"], 2)
    worst = a.stress.loc[a.stress["full_pnl"].idxmin()]
    t["worst_sc"] = worst["scenario"]
    t["worst_pnl"] = smm(worst["full_pnl"])
    t["worst_pct"] = pct(worst["full_pnl_pct_nav"], 2)
    for name, key in (("IG spreads widen (AA+30/A+50/BBB+80)", "cred"), ("Risk-off: rates -50 + IG widening", "riskoff"),
                      ("Stagflation: rates +100 + IG widening", "stag"), ("Bear steepener (2s -25 / 10s+ +25)", "steep"),
                      ("Bull flattener (2s +25 / 10s+ -25)", "flat")):
        row = s.loc[name]
        t[f"{key}_full"] = smm(row["full_pnl"])
        t[f"{key}_rates"] = smm(row["rates_pnl"])
        t[f"{key}_spread"] = smm(row["spread_pnl"])
        t[f"{key}_inter"] = smm(row["interaction_pnl"], 3)
        t[f"{key}_err1_pct"] = pct(abs(row["first_order_error"] / row["full_pnl"]), 2)
        t[f"{key}_err2_pct"] = pct(abs(row["second_order_error"] / row["full_pnl"]), 2)
    t["max_err2_pct"] = pct((a.stress["second_order_error"].abs() / a.stress["full_pnl"].abs()).max(), 2)
    t["max_err1_pct"] = pct((a.stress["first_order_error"].abs() / a.stress["full_pnl"].abs()).max(), 2)
    sa = a.stress_after.set_index("scenario")
    t["r_up100_full"] = smm(sa.loc["Parallel +100bp", "full_pnl"])
    t["r_stag_full"] = smm(sa.loc["Stagflation: rates +100 + IG widening", "full_pnl"])
    t["r_cred_full"] = smm(sa.loc["IG spreads widen (AA+30/A+50/BBB+80)", "full_pnl"])
    cv = a.convergence.set_index("shock_bp")
    for bp in (1, 10, 100):
        t[f"conv{bp}_e1"] = f"${cv.loc[bp, 'abs_err_first']:,.2f}"
        t[f"conv{bp}_e2"] = f"${cv.loc[bp, 'abs_err_second']:,.2f}"
    fit = a.convergence[(a.convergence["shock_bp"] >= 2) & (a.convergence["shock_bp"] <= 50)]
    lx = np.log(fit["shock_bp"])
    t["conv_slope1"] = f"{np.polyfit(lx, np.log(fit['abs_err_first']), 1)[0]:.2f}"
    t["conv_slope2"] = f"{np.polyfit(lx, np.log(fit['abs_err_second']), 1)[0]:.2f}"

    pc = a.pca
    for j in range(3):
        t[f"ev{j + 1}"] = pct(pc.explained[j])
        t[f"pc{j + 1}_label"] = pc.shape_label(j)
        t[f"all_ev{j + 1}"] = pct(a.pca_all_tenors.explained[j])
        t[f"pc{j + 1}_sigma"] = f"{np.sqrt(pc.eigenvalues[j]):.2f}"
        t[f"pc{j + 1}_1s_pnl"] = usd(a.factor_before["one_sigma_daily_pnl"][j])
        t[f"pc{j + 1}_1s_pnl_r"] = usd(a.factor_after["one_sigma_daily_pnl"][j])
        t[f"pc{j + 1}_vshare"] = pct(a.factor_before["variance_share"][j])
        t[f"pc{j + 1}_vshare_r"] = pct(a.factor_after["variance_share"][j])
    t["ev123"] = pct(pc.explained[:3].sum())
    t["all_pc3_1m"] = f"{a.pca_all_tenors.loadings[0, 2]:+.2f}"
    t["pca_n"] = f"{pc.n_obs:,}"
    t["pca_start"], t["pca_end"] = pc.start, pc.end
    t["pc2_short"] = f"{pc.loadings[0, 1]:+.2f}"
    t["pc2_long"] = f"{pc.loadings[-1, 1]:+.2f}"
    t["pc1_min"] = f"{pc.loadings[:, 0].min():.2f}"
    t["pc1_max"] = f"{pc.loadings[:, 0].max():.2f}"
    t["sigma_day"] = usd(a.factor_before["daily_sigma_total"])
    t["sigma_day_r"] = usd(a.factor_after["daily_sigma_total"])
    t["var99"] = usd(a.factor_before["var99_parametric"])
    t["var99_r"] = usd(a.factor_after["var99_parametric"])
    t["hvar99"] = usd(a.hist_var_before["var99_historical"])
    t["hvar99_r"] = usd(a.hist_var_after["var99_historical"])
    t["worst_day"] = usd(a.hist_var_before["worst_day"])
    t["worst_day_date"] = a.hist_var_before["worst_day_date"]
    pcs = a.pca_scenarios.set_index(["scenario", "book"])
    lvl = [i for i in pcs.index if i[0].startswith("PC1") and "+2" in i[0]][0][0]
    slp = [i for i in pcs.index if i[0].startswith("PC2") and "+2" in i[0]][0][0]
    t["pc1_2s"] = smm(pcs.loc[(lvl, "before"), "full_pnl"])
    t["pc1_2s_r"] = smm(pcs.loc[(lvl, "after"), "full_pnl"])
    t["pc2_2s"] = smm(pcs.loc[(slp, "before"), "full_pnl"])
    t["pc2_2s_r"] = smm(pcs.loc[(slp, "after"), "full_pnl"])

    rt = a.rebalance_terms
    pol = rt["problem"].policy
    t["pol_issuer"], t["pol_sector"] = pct(pol.max_issuer, 0), pct(pol.max_sector, 0)
    t["pol_tsy"], t["pol_turn"], t["pol_pos"] = pct(pol.min_treasury, 0), pct(pol.max_turnover, 0), pct(pol.max_position, 0)
    t["turnover"] = pct(rt["turnover"], 2)
    t["cost"] = usd(rt["cost_usd"])
    t["cost_bp"] = f"{rt['cost_usd'] / p.nav * 1e4:.2f}"
    t["n_trades"] = str(rt["n_trades"])
    t["krgap_before"] = f"{rt['before']['krd_l1_gap']:.2f}"
    t["krgap_after"] = f"{rt['after']['krd_l1_gap']:.2f}"
    t["krgap_cut"] = pct(1 - rt["after"]["krd_l1_gap"] / rt["before"]["krd_l1_gap"], 0)
    t["durgap_before"] = f"{rt['before']['duration_gap']:+.2f}"
    t["durgap_after"] = f"{rt['after']['duration_gap']:+.2f}"
    t["n_violations_before"] = str(int((~a.constraints_before["pass"]).sum()))
    t["r_cash"] = usd(r.cash)
    t["ytm_change_bp"] = f"{(r.ytm - p.ytm) * 1e4:+.0f}"
    t["dv01_cut"] = pct(1 - r.dv01 / p.dv01, 0)
    t["cs01_cut"] = pct(1 - r.cs01 / p.cs01, 0)

    # ---- tables -------------------------------------------------------------------
    rows = []
    for label, fn in (("Market value", lambda x: usd(x.nav)), ("Yield (MV-weighted YTM)", lambda x: pct(x.ytm, 2)),
                      ("Effective duration (yrs)", lambda x: f"{x.duration:.2f}"),
                      ("DV01 ($ per bp)", lambda x: usd(x.dv01)), ("Convexity", lambda x: f"{x.convexity:.1f}"),
                      ("CS01 ($ per bp of spread)", lambda x: usd(x.cs01)),
                      ("Spread duration (yrs)", lambda x: f"{x.spread_duration:.2f}"),
                      ("Treasury weight", lambda x: pct(x.treasury_weight)),
                      ("Largest issuer", lambda x: f"{x.group_weights('issuer').index[0]} {pct(x.group_weights('issuer').iloc[0], 2)}"),
                      ("Largest sector", lambda x: f"{x.group_weights('sector').index[0]} {pct(x.group_weights('sector').iloc[0], 2)}")):
        rows.append({"Measure": label, "Portfolio (before)": fn(p), "Benchmark": fn(b), "Rebalanced": fn(r)})
    t["table:risk_summary"] = _md_table(pd.DataFrame(rows))

    krt = pd.DataFrame({"Key rate": TENOR_LABELS, "Portfolio": [usd(v) for v in p.kr_dv01],
                        "Benchmark": [usd(v) for v in b.kr_dv01], "Active": [usd(v) for v in p.kr_dv01 - b.kr_dv01],
                        "Rebalanced": [usd(v) for v in r.kr_dv01]})
    krt.loc[len(krt)] = ["**Sum**", f"**{usd(p.kr_dv01.sum())}**", f"**{usd(b.kr_dv01.sum())}**",
                         f"**{usd((p.kr_dv01 - b.kr_dv01).sum())}**", f"**{usd(r.kr_dv01.sum())}**"]
    t["table:kr"] = _md_table(krt)

    st = a.stress.assign(**{
        "Full repricing": a.stress["full_pnl"].map(smm), "% NAV": a.stress["full_pnl_pct_nav"].map(lambda v: pct(v, 2)),
        "Rates": a.stress["rates_pnl"].map(smm), "Spread": a.stress["spread_pnl"].map(smm),
        "1st-order": a.stress["first_order_pnl"].map(smm), "1st err": a.stress["first_order_error"].map(usd),
        "2nd-order": a.stress["second_order_pnl"].map(smm), "2nd err": a.stress["second_order_error"].map(usd),
        "After rebalance": a.stress_after["full_pnl"].map(smm)})
    t["table:stress"] = _md_table(st[["scenario", "Full repricing", "% NAV", "Rates", "Spread", "1st-order", "1st err",
                                      "2nd-order", "2nd err", "After rebalance"]].rename(columns={"scenario": "Scenario"}))
    ps = a.pca_scenarios.pivot(index="scenario", columns="book", values="full_pnl").reindex(
        a.pca_scenarios["scenario"].drop_duplicates())
    t["table:pca_stress"] = _md_table(pd.DataFrame({"Factor scenario (2 sigma over 21 trading days)": ps.index,
                                                    "Before": ps["before"].map(smm), "After": ps["after"].map(smm)}))

    pt = pd.DataFrame({"Factor": [f"PC{j + 1}" for j in range(3)],
                       "Empirical shape": [pc.shape_label(j) for j in range(3)],
                       "Variance explained": [pct(v) for v in pc.explained[:3]],
                       "Daily sigma (bp along PC)": [f"{np.sqrt(v):.2f}" for v in pc.eigenvalues[:3]],
                       "Portfolio 1-sigma daily P&L": [usd(v) for v in a.factor_before["one_sigma_daily_pnl"]],
                       "Share of portfolio rate variance": [pct(v) for v in a.factor_before["variance_share"]],
                       "After rebalance 1-sigma P&L": [usd(v) for v in a.factor_after["one_sigma_daily_pnl"]]})
    t["table:pca"] = _md_table(pt)

    rows = []
    for (_, bf), (_, af) in zip(a.constraints_before.iterrows(), a.constraints_after.iterrows(), strict=True):
        is_turn = bf["constraint"] == "One-way turnover"
        rows.append({"Constraint": af["constraint"].split(" (")[0],
                     "Limit": ("<= " if af["type"] == "max" else ">= ") + pct(af["limit"], 1),
                     "Before": "-" if is_turn else f"{pct(bf['value'], 2)} {'PASS' if bf['pass'] else 'BREACH'}",
                     "After": f"{pct(af['value'], 2)} {'PASS' if af['pass'] else 'BREACH'}"})
    t["table:constraints"] = _md_table(pd.DataFrame(rows))

    tr = a.trades[a.trades["trade_face"] != 0].sort_values("trade_mv")
    t["table:trades"] = _md_table(pd.DataFrame({
        "Bond": tr["bond_id"], "Side": tr["side"], "Face": tr["trade_face"].map(lambda v: f"{v:+,.0f}"),
        "Market value": tr["trade_mv"].map(usd), "Cost": tr["cost_usd"].map(usd),
        "Weight before": tr["weight_before"].map(lambda v: pct(v, 2)), "Weight after": tr["weight_after"].map(lambda v: pct(v, 2))}))

    sw = pd.DataFrame({"before": p.group_weights("sector"), "after": r.group_weights("sector"),
                       "benchmark": b.group_weights("sector")}).fillna(0.0).sort_values("before", ascending=False)
    t["table:sectors"] = _md_table(pd.DataFrame({"Sector": sw.index, "Before": sw["before"].map(pct),
                                                 "After": sw["after"].map(pct), "Benchmark": sw["benchmark"].map(pct)}))

    hb = held.sort_values(["kind", "years_to_maturity"], ascending=[False, True])
    t["table:holdings"] = _md_table(pd.DataFrame({
        "Bond": hb["bond_id"], "Issuer": hb["issuer"], "Rating": hb["rating"], "Face": hb["face"].map(lambda v: f"{v:,.0f}"),
        "Clean": hb["clean"].map(lambda v: f"{v:.3f}"), "YTM": hb["ytm"].map(lambda v: pct(v, 3)),
        "Z-sprd (bp)": hb["z_spread_bp"].map(lambda v: f"{v:.0f}"), "Eff dur": hb["eff_duration"].map(lambda v: f"{v:.2f}"),
        "DV01 $": hb["dv01"].map(usd), "CS01 $": hb["cs01"].map(usd), "Weight": hb["weight"].map(lambda v: pct(v, 2))}))

    ex = f.set_index("bond_id").loc[["UST 5.250 08/15/36", "UST STRIP 0 11/15/36", "ATLAS 5.875 03/15/36"]]
    t["table:bond_examples"] = _md_table(pd.DataFrame({
        "Bond": ex.index, "Clean": ex["clean"].map(lambda v: f"{v:.4f}"), "Accrued": ex["accrued"].map(lambda v: f"{v:.4f}"),
        "Dirty": ex["dirty"].map(lambda v: f"{v:.4f}"), "YTM": ex["ytm"].map(lambda v: pct(v, 3)),
        "Macaulay": ex["macaulay"].map(lambda v: f"{v:.3f}"), "Modified": ex["modified"].map(lambda v: f"{v:.3f}"),
        "Effective": ex["eff_duration"].map(lambda v: f"{v:.3f}"), "Convexity": ex["eff_convexity"].map(lambda v: f"{v:.1f}"),
        "DV01 /100": ex["dv01"].map(lambda v: f"{v:.4f}"), "Spread dur": ex["spread_duration"].map(lambda v: f"{v:.3f}")}))
    return t


def render(template: str, tokens: dict[str, str]) -> str:
    missing = sorted({m for m in TOKEN.findall(template) if m not in tokens})
    if missing:
        raise KeyError(f"template tokens without values: {missing}")
    return TOKEN.sub(lambda m: tokens[m.group(1)], template)


def render_documents(tokens: dict[str, str], root: Path = ROOT, templates: Path = TEMPLATES_DIR) -> dict[str, str]:
    return {name: render((templates / name).read_text(encoding="utf-8"), tokens) for name in DOCS}


def write_documents(tokens: dict[str, str], root: Path = ROOT) -> list[Path]:
    (RESULTS_DIR / "report_tokens.json").write_text(json.dumps(tokens, indent=1, ensure_ascii=False) + "\n",
                                                   encoding="utf-8", newline="\n")
    paths = []
    for name, text in render_documents(tokens, root).items():
        p = root / name
        p.write_text(text, encoding="utf-8", newline="\n")
        paths.append(p)
    return paths
