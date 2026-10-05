"""Figures for the README / ANALYSIS (matplotlib, Agg backend, deterministic)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from firisk.bootstrap import par_yield  # noqa: E402
from firisk.data import TENOR_LABELS, TENOR_YEARS  # noqa: E402
from firisk.universe import UNIVERSE  # noqa: E402

# Validated categorical order (dataviz reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#ffffff"
BLUE_LIGHT = "#9ec5f4"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "lines.linewidth": 2.0, "svg.hashsalt": "firisk",
})

usd_k = FuncFormatter(lambda v, _: f"${v / 1e3:,.0f}k")
usd_mm = FuncFormatter(lambda v, _: f"${v / 1e6:,.1f}MM")
pct = FuncFormatter(lambda v, _: f"{v:.0%}")


def _save(fig, out: Path, name: str) -> Path:
    p = out / name
    fig.savefig(p, dpi=150, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return p


def _bond(bond_id: str):
    return next(b for b in UNIVERSE if b.bond_id == bond_id)


def fig_cashflows(a, out: Path) -> Path:
    b = _bond("UST 5.250 08/15/36")
    t, cf = b.cashflow_times(a.market.valuation_date)
    pv = cf * a.market.curve.df(t)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    w = 0.18
    ax.bar(t - w / 2, cf, width=w, color=BLUE_LIGHT, label="Promised cash flow")
    ax.bar(t + w / 2, pv, width=w, color=BLUE, label="Present value today")
    ax.annotate(f"Final coupon + $100 principal\nworth only ${pv[-1]:.2f} today",
                xy=(t[-1] + w / 2, pv[-1]), xytext=(t[-1] - 4.2, 70), color=INK2,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.annotate(f"Coupons: ${cf[0]:.3f} every 6 months", xy=(t[2], cf[2]), xytext=(t[2] - 0.3, 22), color=INK2,
                arrowprops={"arrowstyle": "->", "color": MUTED})
    ax.set_title(f"A bond is a list of cash flows: {b.bond_id} (per $100 face)")
    ax.set_xlabel("Years from valuation date")
    ax.set_ylabel("$ per 100 face")
    ax.legend(loc="upper left")
    ax.text(0.5, 0.97, f"Sum of present values = dirty price = ${pv.sum():.3f}", transform=ax.transAxes,
            ha="center", va="top", color=INK)
    return _save(fig, out, "fig01_cashflows.png")


def fig_price_yield(a, out: Path) -> Path:
    b = _bond("UST 5.625 08/15/56")
    vd = a.market.valuation_date
    row = a.risk.frame.set_index("bond_id").loc[b.bond_id]
    y0 = row["ytm"]
    ys = y0 + np.linspace(-0.03, 0.03, 241)
    price = np.array([b.dirty_from_yield(y, vd) for y in ys])
    ya = b.yield_analytics(y0, vd)
    dy = ys - y0
    first = ya.dirty * (1 - ya.modified * dy)
    second = ya.dirty * (1 - ya.modified * dy + 0.5 * ya.convexity * dy**2)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(ys * 100, price, color=BLUE, lw=2.6, label="Exact price (full repricing)")
    ax.plot(ys * 100, first, color=ORANGE, ls="--", label=f"Duration only (straight line, D={ya.modified:.1f})")
    ax.plot(ys * 100, second, color=AQUA, ls=":", lw=2.4, label="Duration + convexity (parabola)")
    ax.scatter([y0 * 100], [ya.dirty], s=60, color=INK, zorder=5, edgecolor=SURFACE, linewidth=2)
    ax.annotate(f"Today: yield {y0:.2%}, price {ya.dirty:.2f}", (y0 * 100, ya.dirty), xytext=(y0 * 100 + 0.4, ya.dirty + 18),
                arrowprops={"arrowstyle": "->", "color": MUTED}, color=INK2)
    ax.set_title(f"Price falls as yield rises - and the curve bends (convexity): {b.bond_id}")
    ax.set_xlabel("Yield to maturity (%)")
    ax.set_ylabel("Dirty price per 100 face")
    ax.legend(loc="upper right")
    return _save(fig, out, "fig02_price_yield.png")


def fig_curves(a, out: Path) -> Path:
    c = a.market.curve
    t = np.linspace(1 / 12, 30, 600)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    fwd_t = np.arange(0, 29.75, 0.25)
    fwd = c.forward_rate(np.maximum(fwd_t, 1e-6), fwd_t + 0.25, "semi")
    ax.step(fwd_t, fwd * 100, where="post", color=ORANGE, lw=1.6, label="3M forward rate")
    ax.plot(t, c.zero_rate(t, "semi") * 100, color=BLUE, lw=2.4, label="Spot (zero-coupon) rate")
    grid = np.arange(1, 61) * 0.5
    ax.plot(grid, [par_yield(c, g) * 100 for g in grid], color=AQUA, lw=1.6, ls="--", label="Model par yield")
    ax.scatter(TENOR_YEARS, a.market.par_yields * 100, s=46, color=INK, zorder=5, edgecolor=SURFACE,
               linewidth=1.8, label="Treasury par yields (H.15 input)")
    ax.set_title(f"U.S. Treasury curves on {a.market.valuation_date:%d %b %Y} (semiannual basis)")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Rate (%)")
    ax.legend(loc="lower right", fontsize=9.5)
    ax2.plot(t, c.df(t), color=BLUE, lw=2.4)
    for tt in (5, 10, 30):
        ax2.scatter([tt], [c.df(tt)], s=40, color=BLUE, edgecolor=SURFACE, linewidth=1.5, zorder=5)
        ax2.annotate(f"${100 * float(c.df(tt)):.1f}", (tt, c.df(tt)), xytext=(4, 6), textcoords="offset points", color=INK2)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("Discount factor DF(t) = value today of $1 at t")
    ax2.set_xlabel("Maturity (years)")
    ax2.set_ylabel("DF(t)")
    fig.tight_layout()
    return _save(fig, out, "fig03_curves.png")


def fig_history(a, out: Path) -> Path:
    h = a.history.frame
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for lab, col in (("2Y", BLUE), ("10Y", ORANGE), ("30Y", AQUA)):
        ax.plot(h.index, h[lab], color=col, lw=1.4, label=lab)
        ax.annotate(lab, (h.index[-1], h[lab].iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, va="center")
    ax.set_title(f"Treasury yields {h.index[0]:%Y}-{h.index[-1]:%Y} ({a.history.source}): the history the PCA learns from")
    ax.set_ylabel("Par yield (%)")
    ax.legend(loc="upper left", ncols=3)
    return _save(fig, out, "fig04_yield_history.png")


def fig_key_rate(a, out: Path) -> Path:
    x = np.arange(len(TENOR_LABELS))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.4))
    for off, p, col in ((-w, a.portfolio, BLUE), (0, a.benchmark, MUTED), (w, a.rebalanced, AQUA)):
        ax.bar(x + off, p.kr_dv01, width=w - 0.03, color=col, label=f"{p.name} (total ${p.dv01 / 1e3:,.1f}k)")
    k = int(np.argmax(a.portfolio.kr_dv01))
    ax.annotate(f"Largest bucket: {TENOR_LABELS[k]}\n${a.portfolio.kr_dv01[k]:,.0f} per bp", (x[k] - w, a.portfolio.kr_dv01[k]),
                xytext=(x[k] - 3.6, a.portfolio.kr_dv01[k] * 0.93), arrowprops={"arrowstyle": "->", "color": MUTED}, color=INK2)
    ax.set_xticks(x, TENOR_LABELS)
    ax.yaxis.set_major_locator(MultipleLocator(4000))
    ax.yaxis.set_major_formatter(usd_k)
    ax.set_title("Key-rate DV01: dollars lost per 1bp rise at each point of the curve")
    ax.set_xlabel("Par-curve key rate")
    ax.set_ylabel("$ per bp")
    ax.legend(loc="upper left")
    return _save(fig, out, "fig05_key_rate_dv01.png")


def fig_stress(a, out: Path) -> Path:
    s = a.stress.iloc[::-1]
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    colors = [AQUA if v >= 0 else RED for v in s["full_pnl"]]
    ax.barh(y, s["full_pnl"], color=colors, height=0.62, label="Full repricing")
    ax.scatter(s["first_order_pnl"], y, marker="|", s=260, color=INK, linewidths=2, label="1st-order (KR-DV01 + CS01)", zorder=5)
    ax.scatter(s["second_order_pnl"], y, marker="o", s=36, color=YELLOW, edgecolor=INK, linewidths=0.8,
               label="2nd-order (+ convexity)", zorder=6)
    for yi, (v, a1, a2) in enumerate(zip(s["full_pnl"], s["first_order_pnl"], s["second_order_pnl"], strict=True)):
        edge = max(v, a1, a2) + 2.5e5 if v >= 0 else min(v, a1, a2) - 2.5e5
        ax.text(edge, y[yi], f"{v / 1e6:+.2f}MM", va="center",
                ha="left" if v >= 0 else "right", color=INK2, fontsize=9)
    ax.set_yticks(y, s["scenario"])
    ax.xaxis.set_major_formatter(usd_mm)
    ax.axvline(0, color=INK2, lw=0.8)
    lim = np.abs(s["full_pnl"]).max() * 1.35
    ax.set_xlim(-lim, lim)
    ax.set_title("Instantaneous stress P&L of the simulated $100MM book")
    ax.set_xlabel("P&L")
    ax.legend(loc="upper right", fontsize=9)
    return _save(fig, out, "fig06_stress_pnl.png")


def fig_attribution(a, out: Path) -> Path:
    s = a.stress[a.stress["family"].isin(["credit", "combined"])]
    x = np.arange(len(s))
    w = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for off, col, c, lab in ((-w, "rates_pnl", BLUE, "Rates"), (0, "spread_pnl", ORANGE, "Credit spread"),
                             (w, "interaction_pnl", MUTED, "Interaction")):
        ax.bar(x + off, s[col], width=w - 0.03, color=c, label=lab)
    ax.scatter(x, s["full_pnl"], marker="D", s=50, color=INK, zorder=5, label="Total (full repricing)")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x, [n.replace(": ", ":\n").replace(" + ", "\n+ ") for n in s["scenario"]], fontsize=9)
    ax.yaxis.set_major_formatter(usd_mm)
    ax.set_title("Rate / spread attribution of the credit scenarios")
    ax.set_ylabel("P&L")
    ax.legend(loc="lower left", ncols=2, fontsize=9)
    return _save(fig, out, "fig07_attribution.png")


def fig_convergence(a, out: Path) -> Path:
    c = a.convergence[a.convergence["shock_bp"] > 0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.loglog(c["shock_bp"], c["abs_err_first"], "o-", color=ORANGE, label="1st-order error (duration only)")
    ax.loglog(c["shock_bp"], c["abs_err_second"], "s-", color=AQUA, label="2nd-order error (+ convexity)")
    x0 = c["shock_bp"].to_numpy()
    for slope, base, ls in ((2, c["abs_err_first"].iloc[0], (0, (2, 2))), (3, c["abs_err_second"].iloc[2] / x0[2] ** 3, (0, (2, 2)))):
        ref = base * x0**slope if slope == 3 else base * (x0 / x0[0]) ** 2
        ax.loglog(x0, ref, color=MUTED, lw=1, ls=ls)
        ax.text(x0[-1] * 1.08, ref[-1], f"slope {slope}", color=MUTED, va="center")
    ax.set_title("Approximation error vs full repricing (parallel shocks)")
    ax.set_xlabel("Shock size (bp, log scale)")
    ax.set_ylabel("|error| in $ (log scale)")
    ax.legend(loc="upper left")
    return _save(fig, out, "fig08_approx_convergence.png")


def fig_pca(a, out: Path) -> Path:
    p = a.pca
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2, 1]})
    for j, col in zip(range(3), (BLUE, ORANGE, AQUA), strict=True):
        ax.plot(p.tenor_years, p.loadings[:, j], "o-", color=col, ms=6,
                label=f"PC{j + 1} = {p.shape_label(j)} ({p.explained[j]:.1%})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks(p.tenor_years, p.tenor_labels)
    ax.minorticks_off()
    ax.set_title(f"PCA loadings of daily par-curve changes, {p.start[:4]}-{p.end[:4]}")
    ax.set_xlabel("Tenor (log scale)")
    ax.set_ylabel("Loading")
    ax.legend(loc="lower right")
    ev = p.explained[:5]
    ax2.bar([f"PC{i + 1}" for i in range(5)], ev, color=[BLUE, ORANGE, AQUA, MUTED, MUTED])
    for i, v in enumerate(ev):
        ax2.text(i, v + 0.01, f"{v:.1%}", ha="center", color=INK2, fontsize=9)
    ax2.yaxis.set_major_formatter(pct)
    ax2.set_title("Variance explained")
    fig.tight_layout()
    return _save(fig, out, "fig09_pca_loadings.png")


def fig_composition(a, out: Path) -> Path:
    pol = a.rebalance_terms["problem"].policy
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for axis, col, limit, title in ((ax, "sector", pol.max_sector, "Corporate sector weight"),
                                    (ax2, "issuer", pol.max_issuer, "Corporate issuer weight")):
        before = a.portfolio.group_weights(col)
        after = a.rebalanced.group_weights(col).reindex(before.index).fillna(0.0)
        y = np.arange(len(before))[::-1]
        axis.barh(y + 0.2, before, height=0.38, color=BLUE, label="Before")
        axis.barh(y - 0.2, after, height=0.38, color=AQUA, label="After rebalance")
        axis.axvline(limit, color=RED, lw=1.5, ls="--")
        axis.text(limit, y[0] + 0.9, f" limit {limit:.0%}", color=RED, va="bottom")
        axis.set_yticks(y, before.index, fontsize=9)
        axis.xaxis.set_major_locator(MultipleLocator(0.04 if col == "sector" else 0.01))
        axis.xaxis.set_major_formatter(pct)
        axis.set_title(title)
        axis.legend(loc="lower right")
    tsy_b, tsy_a = a.portfolio.treasury_weight, a.rebalanced.treasury_weight
    fig.suptitle(f"Concentration vs policy limits (Treasuries {tsy_b:.1%} -> {tsy_a:.1%}, minimum {pol.min_treasury:.0%})",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "fig10_composition.png")


def fig_ladder(a, out: Path) -> Path:
    lb, la = a.portfolio.ladder(), a.rebalanced.ladder()
    x = np.arange(len(lb))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for off, lad, alpha, name in ((-w / 2, lb, 1.0, "Before"), (w / 2, la, 0.55, "After")):
        ax.bar(x + off, lad["treasury_mv"], width=w - 0.03, color=BLUE, alpha=alpha, label=f"{name}: Treasury")
        ax.bar(x + off, lad["corporate_mv"], bottom=lad["treasury_mv"], width=w - 0.03, color=ORANGE, alpha=alpha,
               label=f"{name}: Corporate", edgecolor=SURFACE, linewidth=1.5)
    ax.set_xticks(x, lb.index.astype(str))
    ax.yaxis.set_major_formatter(usd_mm)
    ax.set_title("Maturity ladder: market value by years to maturity (left bar before, right bar after)")
    ax.set_xlabel("Years to maturity")
    ax.legend(ncols=2, fontsize=9, loc="upper left")
    return _save(fig, out, "fig11_ladder.png")


ALL_FIGURES = [fig_cashflows, fig_price_yield, fig_curves, fig_history, fig_key_rate, fig_stress,
               fig_attribution, fig_convergence, fig_pca, fig_composition, fig_ladder]


def make_all(a, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    return [fn(a, out) for fn in ALL_FIGURES]

