#!/usr/bin/env python3
"""
anthropic_ipo_exercises.py
==========================
Reference solutions to Exercises 1–3 of the Anthropic IPO valuation
learning module.


Exercise 1 — Sensitivity table

    Write a class SensitivityAnalyzer that produces a 5×5 table of value_per_share for initial_wacc ∈ [7%, 8%, 9%, 10%, 11%] and stable_growth ∈ [3%, 4%, 5%, 6%, 7%]. Plot it as a heatmap.
    Hint: reuse ValuationInputs(**{**inp.__dict__, "initial_wacc": w}). Guard rows where stable_wacc <= g — replace with nan.

Exercise 2 — Monte Carlo on levers

    Simulate 1,000 runs drawing growth_y2_5 ~ N(0.42, 0.08), target_margin ~ N(0.325, 0.05), initial_wacc ~ N(0.0974, 0.01) (clip to valid ranges). Report median, 5th, and 95th percentile of value_per_share.
    Hint: use np.random.default_rng(42) and wrap each run in try/except ValueError. Plot the histogram.

Exercise 3 — Bull / base / bear scenarios

    Build three ValuationInputs instances: bull (growth=0.55, margin=0.40, wacc=0.09), base (canonical), bear (growth=0.25, margin=0.20, wacc=0.11). Produce a single grouped bar chart of value_per_share across the three, with the current price of $525 drawn as a horizontal dashed line.
    Hint: if the base case is below $525, your story is bearish; if the bear case is above $525, the market is pricing in something the DCF doesn't support.




Design notes
------------
- Every solution is a `@dataclass` so the learner can instantiate it,
  inspect it, and reuse it in their own notebooks.
- All three solutions reuse the same `DCFEngine` and `ValuationInputs`
  from the main module (`from anthropic_ipo_valuation import ...`), so
  there is no duplicated DCF math.
- Fixed seed `SEED=42` guarantees reproducibility of the Monte Carlo.

Usage
-----
    python3 anthropic_ipo_exercises.py

Produces:
    fig_ex1_heatmap.png    - WACC × growth sensitivity heatmap
    fig_ex2_hist.png       - Monte-Carlo distribution of per-share value
    fig_ex3_scenarios.png  - bull / base / bear bar chart

Dependencies
------------
Python 3.10+, numpy, pandas, matplotlib.  Also requires the main
module `anthropic_ipo_valuation.py` in the same directory.

Source datasets
---------------
- Synthetic Anthropic inputs: `build_canonical_anthropic_inputs()` in the
  main module (seed = 42).
- Damodaran (Jan 2026) industry averages used for plausibility checks
  only, not embedded.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from anthropic_ipo_valuation import (
    DCFEngine,
    LeaseConverter,
    RndConverter,
    ValuationInputs,
    build_canonical_anthropic_inputs,
)

SEED = 42
np.random.seed(SEED)


# ======================================================================
# EXERCISE 1 — Sensitivity table
# ======================================================================

@dataclass
class SensitivityAnalyzer:
    """
    Build a 2-D grid of `value_per_share` across WACC and terminal growth.

    Attributes
    ----------
    base_inputs : ValuationInputs
    rd : RndConverter
    lease : LeaseConverter
    wacc_values : list of float   (fractions, e.g. 0.07)
    growth_values : list of float
    """

    base_inputs: ValuationInputs
    rd: RndConverter
    lease: LeaseConverter
    wacc_values: List[float] = field(
        default_factory=lambda: [0.07, 0.08, 0.09, 0.10, 0.11]
    )
    growth_values: List[float] = field(
        default_factory=lambda: [0.03, 0.04, 0.05, 0.06, 0.07]
    )

    def run(self) -> pd.DataFrame:
        """
        Grid of per-share values.

        Cells where `stable_wacc <= g` are NaN — the DCF is undefined
        (Gordon growth divides by zero or a negative number).
        """
        rows = []
        for w in self.wacc_values:
            row = {"initial_wacc": w}
            for g in self.growth_values:
                if self.base_inputs.stable_wacc <= g:
                    row[f"g={g:.0%}"] = float("nan")
                    continue
                try:
                    modified = replace(
                        self.base_inputs,
                        initial_wacc=w,
                        stable_growth=g,
                    )
                    r = DCFEngine(modified, self.rd, self.lease).run()
                    row[f"g={g:.0%}"] = r.value_per_share
                except ValueError:
                    row[f"g={g:.0%}"] = float("nan")
            rows.append(row)
        return pd.DataFrame(rows).set_index("initial_wacc")

    def plot(self, df: pd.DataFrame, path: str = "fig_ex1_heatmap.png") -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(df.values, aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(df.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels([f"{i:.0%}" for i in df.index])
        ax.set_xlabel("Terminal growth (stable_growth)")
        ax.set_ylabel("Initial WACC")
        ax.set_title("Exercise 1 — Per-share value sensitivity\n"
                     "(price = $525 dashed if in range)")
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                val = df.values[i, j]
                if np.isnan(val):
                    continue
                ax.text(j, i, f"${val:,.0f}", ha="center", va="center",
                        fontsize=8, color="black")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Value per share ($)")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex1] Saved {path}")


# ======================================================================
# EXERCISE 2 — Monte Carlo on the levers
# ======================================================================

@dataclass
class MonteCarloAnalyzer:
    """
    Simulate N draws of the three most influential levers and report
    distribution statistics of `value_per_share`.

    Levers are sampled from Normal distributions clipped to valid ranges:
    - growth_y2_5 : N(0.42, 0.08) clipped to [0, 1.5]
    - target_margin : N(0.325, 0.05) clipped to [0.05, 0.60]
    - initial_wacc : N(0.0974, 0.01) clipped to [stable_growth+0.005, 0.20]
    """

    base_inputs: ValuationInputs
    rd: RndConverter
    lease: LeaseConverter
    n_runs: int = 1_000
    seed: int = SEED

    def run(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        samples = {
            "growth_y2_5": np.clip(rng.normal(0.42, 0.08, self.n_runs), 0.0, 1.5),
            "target_margin": np.clip(rng.normal(0.325, 0.05, self.n_runs), 0.05, 0.60),
            "initial_wacc": np.clip(
                rng.normal(0.0974, 0.01, self.n_runs),
                self.base_inputs.stable_growth + 0.005,
                0.20,
            ),
        }
        values: List[float] = []
        for i in range(self.n_runs):
            try:
                modified = replace(
                    self.base_inputs,
                    growth_y2_5=float(samples["growth_y2_5"][i]),
                    target_margin=float(samples["target_margin"][i]),
                    initial_wacc=float(samples["initial_wacc"][i]),
                )
                r = DCFEngine(modified, self.rd, self.lease).run()
                values.append(r.value_per_share)
            except (ValueError, ZeroDivisionError):
                # Skip invalid combinations; could log them for audit.
                continue
        return pd.DataFrame({"value_per_share": values})

    def plot(self, df: pd.DataFrame, path: str = "fig_ex2_hist.png") -> None:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(df["value_per_share"], bins=40, color="steelblue", edgecolor="black")
        median = df["value_per_share"].median()
        p5 = df["value_per_share"].quantile(0.05)
        p95 = df["value_per_share"].quantile(0.95)
        for v, label, color in [
            (median, f"median = ${median:,.0f}", "darkorange"),
            (p5, f"5th pct = ${p5:,.0f}", "indianred"),
            (p95, f"95th pct = ${p95:,.0f}", "indianred"),
        ]:
            ax.axvline(v, color=color, ls="--", label=label)
        ax.set_xlabel("Value per share ($)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Exercise 2 — Monte-Carlo distribution "
                     f"(N={len(df):,} valid runs)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex2] Saved {path}")


# ======================================================================
# EXERCISE 3 — Bull / base / bear scenarios
# ======================================================================

@dataclass
class ScenarioAnalyzer:
    """
    Compare three narratives against the current $525 price.
    """

    base_inputs: ValuationInputs
    rd: RndConverter
    lease: LeaseConverter
    current_price: float = 525.0

    def build_scenarios(self) -> Dict[str, ValuationInputs]:
        bull = replace(
            self.base_inputs,
            growth_y2_5=0.55,
            target_margin=0.40,
            initial_wacc=0.09,
        )
        bear = replace(
            self.base_inputs,
            growth_y2_5=0.25,
            target_margin=0.20,
            initial_wacc=0.11,
        )
        return {"Bull": bull, "Base": self.base_inputs, "Bear": bear}

    def run(self) -> pd.DataFrame:
        rows = []
        for name, inp in self.build_scenarios().items():
            r = DCFEngine(inp, self.rd, self.lease).run()
            rows.append({
                "scenario": name,
                "value_per_share": r.value_per_share,
                "equity_value": r.equity_value,
                "upside_vs_price": (r.value_per_share / self.current_price - 1),
            })
        return pd.DataFrame(rows)

    def plot(self, df: pd.DataFrame, path: str = "fig_ex3_scenarios.png") -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = {"Bull": "seagreen", "Base": "steelblue", "Bear": "indianred"}
        bars = ax.bar(
            df["scenario"],
            df["value_per_share"],
            color=[colors[s] for s in df["scenario"]],
        )
        ax.axhline(
            self.current_price, color="black", ls="--",
            label=f"Current price = ${self.current_price:.0f}",
        )
        for b, v in zip(bars, df["value_per_share"]):
            ax.text(
                b.get_x() + b.get_width() / 2, v,
                f"${v:,.0f}", ha="center", va="bottom",
            )
        ax.set_ylabel("Value per share ($)")
        ax.set_title("Exercise 3 — Bull / Base / Bear vs current price")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Ex3] Saved {path}")


# ======================================================================
# Driver
# ======================================================================

def main() -> None:
    inp, rd, lease = build_canonical_anthropic_inputs()

    print("=" * 78)
    print(" Exercise solutions: Anthropic IPO DCF")
    print("=" * 78)

    # ---------------- Exercise 1 ----------------
    print("\n[Exercise 1] WACC × growth sensitivity")
    ex1 = SensitivityAnalyzer(inp, rd, lease)
    df1 = ex1.run()
    print(df1.round(0).to_string())
    ex1.plot(df1)

    # ---------------- Exercise 2 ----------------
    print("\n[Exercise 2] Monte Carlo on growth, margin, WACC")
    ex2 = MonteCarloAnalyzer(inp, rd, lease, n_runs=1_000)
    df2 = ex2.run()
    stats = df2["value_per_share"].describe(
        percentiles=[0.05, 0.50, 0.95]
    )
    print(stats.round(2).to_string())
    ex2.plot(df2)

    # ---------------- Exercise 3 ----------------
    print("\n[Exercise 3] Bull / base / bear vs $525")
    ex3 = ScenarioAnalyzer(inp, rd, lease)
    df3 = ex3.run()
    print(df3.round(3).to_string(index=False))
    ex3.plot(df3)

    print("\nDone. Figures: fig_ex1_heatmap.png, fig_ex2_hist.png, "
          "fig_ex3_scenarios.png")


if __name__ == "__main__":
    main()
