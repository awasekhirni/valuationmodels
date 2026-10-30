"""
Author: Awase Khirni Syed
Copyright 2025 β ORI Inc.Canada All Rights Reserved.
Thanks to Dr. Aswath Damodaran

pre_ipo_dcf_valuation.py
========================
Educational and institutional-grade DCF fair-value engine for a pre-IPO
AI company (Anthropic).  The module is intentionally self-contained so a
beginner can run one file and see the entire valuation workflow.

Purpose
-------
- Build a bottom-up FCFF (Free Cash Flow to the Firm) DCF from explicit
  assumptions about growth, margins, reinvestment, and risk.
- Reverse-engineer the assumptions implied by an IPO price.
- Stress-test the valuation with scenarios, sensitivity grids, and Monte
  Carlo simulation.
- Teach the mechanics with an ELI5 walkthrough and exercise hints.

Design principles
-----------------
- **DRY**: all discounting, terminal-value, and validation logic lives in
  one place (`DCFModel`).  Scenario, sensitivity, and Monte Carlo engines
  call the same core.
- **Dataclasses**: every stateful concept (`ValuationAssumptions`,
  `DCFResult`, `ScenarioSet`, etc.) is a dataclass, making instantiation
  and inspection trivial.
- **Documentation**: module, class, and method docstrings; inline
  comments explain *why*, not just *what*.
- **Reproducibility**: every stochastic component uses a fixed seed.

Installation
------------
    python3 -m venv .venv
    source .venv/bin/activate
    pip3 install numpy matplotlib
    # Optional: pip3 install pandas  (only for prettier tables)
    python3 pre_ipo_dcf_valuation.py

Outputs
-------
- `fig_fcff_path.png`          : revenue, EBIT, FCFF over the forecast.
- `fig_sensitivity.png`        : equity value heatmap (WACC vs terminal growth).
- `fig_monte_carlo.png`        : distribution of equity values vs IPO price.
- `fig_scenario_comparison.png`: base / bull / bear bar chart.
- Console tables with expected values.

Author: Senior AI Engineer / Educator
Copyright 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS — reported facts and market snapshots
# ---------------------------------------------------------------------------

# Anthropic reported financials (S-1 extracts).
HISTORICAL_REVENUE: Dict[str, float] = {
    "FY2023": 41e6,
    "FY2024": 386e6,
    "FY2025": 4_585e6,
    "H1_2025": 1_163e6,
    "H1_2026": 16_286e6,
}

# Adjusted operating margin reported for H1 2026 (non-GAAP).
CURRENT_ADJ_OPERATING_MARGIN: float = 0.10

# Cash balance reported in the S-1.
REPORTED_CASH: float = 114e9

# Comparable P/S snapshots (mid-to-late 2026).
COMPARABLE_PS: Dict[str, float] = {
    "NVIDIA": 19.0,
    "Microsoft": 9.0,   # midpoint of the 9–12 range
    "SpaceX (private proxy)": 16.0,
}

# IPO price under test.
IPO_EQUITY_VALUE: float = 1.75e12

# Damodaran base-case equity value, used as a calibration anchor.
DAMODARAN_BASE_EQUITY: float = 1.475e12

# ---------------------------------------------------------------------------
# ASSUMPTIONS DATACLASS
# ---------------------------------------------------------------------------


@dataclass
class ValuationAssumptions:
    """
    Every editable input to the DCF.

    Parameters
    ----------
    base_revenue : float
        FY2026E revenue in dollars.  Default $50B is an annualisation of
        the H1 2026 run-rate, not a company forecast.
    growth_rates : list of float
        Year-on-year revenue growth for each explicit forecast year.
        Length defines the explicit forecast horizon.
    current_margin : float
        Starting adjusted operating margin (H1 2026 ≈ 10%).
    target_margin : float
        Margin the company converges to by ``margin_convergence_year``.
    margin_convergence_year : int
        1-indexed year in which ``target_margin`` is reached.
    sales_to_capital : float
        Revenue generated per dollar of invested capital.  A higher ratio
        means less reinvestment is needed for a given growth rate.
    wacc : float
        Weighted average cost of capital (discount rate).
    terminal_growth : float
        Perpetual growth rate after the explicit forecast.
    cash : float
        Cash and marketable securities (added to enterprise value).
    debt : float
        Total debt (subtracted from enterprise value).
    shares_outstanding : float
        Diluted shares outstanding, used for per-share values.
    tax_rate : float
        Marginal corporate tax rate.
    """

    base_revenue: float = 50e9
    growth_rates: List[float] = field(
        default_factory=lambda: [
            0.60, 0.45, 0.35, 0.28, 0.22,
            0.18, 0.14, 0.11, 0.09, 0.07,
            0.06, 0.05, 0.04, 0.03, 0.03,
        ]
    )
    current_margin: float = CURRENT_ADJ_OPERATING_MARGIN
    target_margin: float = 0.40
    margin_convergence_year: int = 8
    sales_to_capital: float = 2.0
    wacc: float = 0.09
    terminal_growth: float = 0.035
    cash: float = REPORTED_CASH
    debt: float = 0.0
    shares_outstanding: float = 1.0e9
    tax_rate: float = 0.21

    def validate(self) -> None:
        """
        Raise ``ValueError`` if any input is economically or
        mathematically invalid.  Called automatically by ``DCFModel``.
        """
        if self.base_revenue <= 0:
            raise ValueError("base_revenue must be positive.")
        if not self.growth_rates:
            raise ValueError("growth_rates must contain at least one year.")
        if self.sales_to_capital <= 0:
            raise ValueError("sales_to_capital must be positive.")
        if not (0.0 <= self.current_margin <= 1.0):
            raise ValueError("current_margin must be between 0 and 1.")
        if not (0.0 <= self.target_margin <= 1.0):
            raise ValueError("target_margin must be between 0 and 1.")
        if self.margin_convergence_year < 1:
            raise ValueError("margin_convergence_year must be >= 1.")
        if self.wacc <= self.terminal_growth:
            raise ValueError(
                "WACC must be greater than terminal_growth; "
                "otherwise the terminal value is infinite or negative."
            )
        if self.shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be positive.")
        if not (0.0 <= self.tax_rate < 1.0):
            raise ValueError("tax_rate must be in [0, 1).")
        if self.cash < 0 or self.debt < 0:
            raise ValueError("cash and debt cannot be negative.")


# ---------------------------------------------------------------------------
# RESULT DATACLASS
# ---------------------------------------------------------------------------


@dataclass
class DCFResult:
    """Container for every intermediate and final DCF output."""

    years: List[int]
    revenue: List[float]
    ebit: List[float]
    nopat: List[float]
    reinvestment: List[float]
    fcff: List[float]
    pv_fcff: List[float]
    terminal_value: float
    pv_terminal_value: float
    enterprise_value: float
    equity_value: float
    value_per_share: float
    assumptions: ValuationAssumptions


# ---------------------------------------------------------------------------
# CORE DCF MODEL
# ---------------------------------------------------------------------------


class DCFModel:
    """
    A single, reusable FCFF DCF engine.

    All scenario, sensitivity, and Monte Carlo analyses call this class;
    no discounting logic is duplicated elsewhere.
    """

    def __init__(self, assumptions: ValuationAssumptions):
        assumptions.validate()
        self.a = assumptions

    # ------------------------------------------------------------------
    # Projection primitives
    # ------------------------------------------------------------------
    def _margin_path(self) -> List[float]:
        """
        Linearly interpolate the operating margin from
        ``current_margin`` to ``target_margin`` over
        ``margin_convergence_year`` years, then hold it flat.
        """
        n = len(self.a.growth_rates)
        path: List[float] = []
        for t in range(1, n + 1):
            if t <= self.a.margin_convergence_year:
                frac = t / self.a.margin_convergence_year
                margin = self.a.current_margin + frac * (
                    self.a.target_margin - self.a.current_margin
                )
            else:
                margin = self.a.target_margin
            path.append(margin)
        return path

    def project(self) -> DCFResult:
        """
        Build the full FCFF projection and discount it to today.

        Returns
        -------
        DCFResult
        """
        a = self.a
        n = len(a.growth_rates)
        years = list(range(1, n + 1))

        # --- Revenue path -------------------------------------------------
        revenue: List[float] = []
        rev = a.base_revenue
        for g in a.growth_rates:
            rev = rev * (1.0 + g)
            revenue.append(rev)

        # --- Margin and EBIT ---------------------------------------------
        margins = self._margin_path()
        ebit = [rev * m for rev, m in zip(revenue, margins)]
        nopat = [e * (1.0 - a.tax_rate) for e in ebit]

        # --- Reinvestment ------------------------------------------------
        # Reinvestment = ΔRevenue / (Sales / Capital).
        # This is the standard "sales-to-capital" approach: growth is not
        # free; each incremental dollar of revenue consumes capital.
        reinvestment: List[float] = []
        prev_rev = a.base_revenue
        for rev in revenue:
            delta_rev = rev - prev_rev
            reinvestment.append(delta_rev / a.sales_to_capital)
            prev_rev = rev

        # --- FCFF ---------------------------------------------------------
        fcff = [n - r for n, r in zip(nopat, reinvestment)]

        # --- Discounting -------------------------------------------------
        pv_fcff = [
            cf / ((1.0 + a.wacc) ** t) for t, cf in zip(years, fcff)
        ]

        # --- Terminal value ----------------------------------------------
        # Gordon growth on the *next* year's FCFF (year n+1).
        fcff_next = fcff[-1] * (1.0 + a.terminal_growth)
        tv = fcff_next / (a.wacc - a.terminal_growth)
        pv_tv = tv / ((1.0 + a.wacc) ** n)

        # --- Enterprise and equity value ---------------------------------
        ev = sum(pv_fcff) + pv_tv
        equity = ev + a.cash - a.debt
        per_share = equity / a.shares_outstanding

        return DCFResult(
            years=years,
            revenue=revenue,
            ebit=ebit,
            nopat=nopat,
            reinvestment=reinvestment,
            fcff=fcff,
            pv_fcff=pv_fcff,
            terminal_value=tv,
            pv_terminal_value=pv_tv,
            enterprise_value=ev,
            equity_value=equity,
            value_per_share=per_share,
            assumptions=a,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------
    def enterprise_value(self) -> float:
        """Return only the enterprise value (avoids building a full result)."""
        return self.project().enterprise_value

    def equity_value(self) -> float:
        """Return only the equity value."""
        return self.project().equity_value


# ---------------------------------------------------------------------------
# SCENARIO ANALYSIS
# ---------------------------------------------------------------------------


@dataclass
class ScenarioSet:
    """
    Base, bull, and bear assumption sets.

    Each scenario is a full ``ValuationAssumptions`` object, so the
    learner can inspect every input side by side.
    """

    base: ValuationAssumptions = field(default_factory=ValuationAssumptions)
    bull: ValuationAssumptions = field(
        default_factory=lambda: ValuationAssumptions(
            growth_rates=[
                0.80, 0.60, 0.45, 0.35, 0.28,
                0.22, 0.18, 0.14, 0.12, 0.10,
                0.08, 0.06, 0.05, 0.04, 0.03,
            ],
            target_margin=0.45,
            margin_convergence_year=7,
            sales_to_capital=2.5,
            wacc=0.085,
        )
    )
    bear: ValuationAssumptions = field(
        default_factory=lambda: ValuationAssumptions(
            growth_rates=[
                0.40, 0.30, 0.20, 0.14, 0.10,
                0.08, 0.06, 0.05, 0.04, 0.03,
                0.03, 0.03, 0.03, 0.03, 0.03,
            ],
            target_margin=0.20,
            margin_convergence_year=10,
            sales_to_capital=1.2,
            wacc=0.11,
        )
    )

    def run(self) -> Dict[str, DCFResult]:
        """Return one ``DCFResult`` per scenario."""
        return {
            name: DCFModel(getattr(self, name)).project()
            for name in ("base", "bull", "bear")
        }


# ---------------------------------------------------------------------------
# SENSITIVITY ANALYSIS
# ---------------------------------------------------------------------------


@dataclass
class SensitivityGrid:
    """
    Equity-value sensitivity to WACC and terminal growth.

    The grid is deterministic: each cell re-runs the full DCF with the
    same assumptions except for the two varied inputs.
    """

    base: ValuationAssumptions = field(default_factory=ValuationAssumptions)
    wacc_range: Tuple[float, float, int] = (0.07, 0.12, 11)
    growth_range: Tuple[float, float, int] = (0.02, 0.05, 7)

    def run(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        waccs : np.ndarray
        growths : np.ndarray
        equity_grid : np.ndarray, shape (len(growths), len(waccs))
        """
        waccs = np.linspace(*self.wacc_range)
        growths = np.linspace(*self.growth_range)
        grid = np.zeros((len(growths), len(waccs)))

        for i, g in enumerate(growths):
            for j, w in enumerate(waccs):
                # Deep copy the base assumptions so mutations do not leak.
                a = ValuationAssumptions(
                    **{
                        **self.base.__dict__,
                        "wacc": float(w),
                        "terminal_growth": float(g),
                    }
                )
                grid[i, j] = DCFModel(a).equity_value()
        return waccs, growths, grid


# ---------------------------------------------------------------------------
# MONTE CARLO SIMULATION
# ---------------------------------------------------------------------------


@dataclass
class MonteCarloSimulator:
    """
    Draw key assumptions from distributions and run the DCF many times.

    Parameters
    ----------
    base : ValuationAssumptions
        Central assumptions.  Random draws are multipliers around these.
    n_simulations : int
        Number of DCF runs.
    seed : int
        Fixed seed for reproducibility.
    """

    base: ValuationAssumptions = field(default_factory=ValuationAssumptions)
    n_simulations: int = 10_000
    seed: int = 42

    def run(self) -> np.ndarray:
        """
        Returns
        -------
        np.ndarray
            Simulated equity values, shape ``(n_simulations,)``.
        """
        rng = np.random.default_rng(self.seed)
        results = np.empty(self.n_simulations)

        for i in range(self.n_simulations):
            # Growth multiplier: log-normal keeps growth positive.
            growth_mult = float(rng.lognormal(mean=0.0, sigma=0.20))
            # Margin: beta-like draw bounded between 5% and 55%.
            margin = float(rng.beta(2.0, 5.0) * 0.50 + 0.05)
            # WACC: normal, truncated implicitly by validation.
            wacc = float(np.clip(rng.normal(0.09, 0.012), 0.06, 0.15))
            # Terminal growth: normal, below WACC.
            tg = float(np.clip(rng.normal(0.035, 0.005), 0.01, 0.05))

            # Skip invalid combinations (WACC <= terminal growth).
            if wacc <= tg:
                continue

            a = ValuationAssumptions(
                **{
                    **self.base.__dict__,
                    "growth_rates": [
                        g * growth_mult for g in self.base.growth_rates
                    ],
                    "target_margin": margin,
                    "wacc": wacc,
                    "terminal_growth": tg,
                }
            )
            results[i] = DCFModel(a).equity_value()

        # Remove any zero-initialised invalid runs.
        return results[results != 0.0]


# ---------------------------------------------------------------------------
# VISUALISER
# ---------------------------------------------------------------------------


class Visualizer:
    """All plotting logic in one place, so the numerical code stays clean."""

    @staticmethod
    def plot_fcff_path(result: DCFResult, save_to: str = "fig_fcff_path.png") -> None:
        """Revenue, EBIT, and FCFF over the explicit forecast."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(result.years, np.array(result.revenue) / 1e9,
                 marker="o", label="Revenue ($B)", color="steelblue")
        ax1.plot(result.years, np.array(result.ebit) / 1e9,
                 marker="s", label="EBIT ($B)", color="darkorange")
        ax1.set_xlabel("Forecast year")
        ax1.set_ylabel("$ billions")
        ax1.set_title("Revenue and EBIT")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        colors = ["seagreen" if x >= 0 else "firebrick" for x in result.fcff]
        ax2.bar(result.years, np.array(result.fcff) / 1e9, color=colors)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_xlabel("Forecast year")
        ax2.set_ylabel("FCFF ($B)")
        ax2.set_title("Free Cash Flow to the Firm")
        ax2.grid(True, axis="y", alpha=0.3)

        fig.suptitle("DCF Projection: Base Case", fontsize=14)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Viz] Saved {save_to}")

    @staticmethod
    def plot_sensitivity(
        waccs: np.ndarray,
        growths: np.ndarray,
        grid: np.ndarray,
        save_to: str = "fig_sensitivity.png",
    ) -> None:
        """Heatmap of equity value ($T) across WACC and terminal growth."""
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            grid / 1e12,
            aspect="auto",
            origin="lower",
            extent=[waccs[0], waccs[-1], growths[0], growths[-1]],
            cmap="RdYlGn",
        )
        ax.set_xlabel("WACC")
        ax.set_ylabel("Terminal growth rate")
        ax.set_title("Equity Value ($ trillions): WACC vs Terminal Growth")
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Equity value ($T)")
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Viz] Saved {save_to}")

    @staticmethod
    def plot_monte_carlo(
        equity_values: np.ndarray,
        ipo_price: float = IPO_EQUITY_VALUE,
        save_to: str = "fig_monte_carlo.png",
    ) -> None:
        """Histogram of simulated equity values with IPO price line."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(equity_values / 1e12, bins=60, color="steelblue",
                edgecolor="white", alpha=0.8)
        ax.axvline(ipo_price / 1e12, color="firebrick", linestyle="--",
                   linewidth=2, label=f"IPO price = ${ipo_price / 1e12:.2f}T")
        ax.set_xlabel("Equity value ($ trillions)")
        ax.set_ylabel("Frequency")
        ax.set_title("Monte Carlo: Distribution of DCF Equity Values")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Viz] Saved {save_to}")

    @staticmethod
    def plot_scenarios(
        scenarios: Dict[str, DCFResult],
        save_to: str = "fig_scenario_comparison.png",
    ) -> None:
        """Bar chart comparing base, bull, and bear equity values."""
        names = list(scenarios.keys())
        values = [scenarios[n].equity_value / 1e12 for n in names]
        colors = ["steelblue", "seagreen", "firebrick"]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, values, color=colors[: len(names)])
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05, f"${v:.2f}T",
                    ha="center", fontsize=11)
        ax.axhline(IPO_EQUITY_VALUE / 1e12, color="black",
                   linestyle="--", linewidth=1.5, label="IPO price")
        ax.set_ylabel("Equity value ($ trillions)")
        ax.set_title("Scenario Comparison: Base, Bull, Bear")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_to, dpi=150)
        plt.show()
        print(f"[Viz] Saved {save_to}")


# ---------------------------------------------------------------------------
# ELI5 WALKTHROUGH
# ---------------------------------------------------------------------------
# Summary (ELI5)

# Imagine you are offered a lemonade stand that will not open for another ten years.
# The seller says it will one day sell millions of cups, but today you must decide what it is worth.
# You cannot use today’s pocket money as a guide; instead, you guess how many cups it will sell each year,
# how much profit each cup makes, how much you must spend on new juicers to grow, and how risky the whole plan feels. You then “discount” every future dollar back to today, because a dollar ten years from now is worth less than a dollar in your hand. If the seller’s price is higher than the pile of today’s dollars you just calculated, you walk away. That is a DCF (Discounted Cash Flow) valuation: a bottom-up estimate of what a business is worth today, built from explicit beliefs about growth, margins, reinvestment, and risk — not from what a comparable company happens to trade at


def eli5_walkthrough() -> None:
    """
    A toy DCF with plain-language print statements, suitable for a
    complete beginner.  Uses three years and small round numbers so the
    arithmetic is easy to follow by hand.
    """
    print("\n" + "=" * 70)
    print("ELI5 WALKTHROUGH: How a DCF works")
    print("=" * 70)

    revenue = 100.0
    growth = [0.50, 0.30, 0.10]
    margin = [0.10, 0.20, 0.30]
    sales_to_capital = 2.0
    tax = 0.21
    wacc = 0.10
    terminal_growth = 0.03

    print("\nStep 1: Start with today's revenue.")
    print(f"  Revenue today = ${revenue:.0f}")

    revenues, ebits, fcffs = [], [], []
    prev_rev = revenue
    for t, (g, m) in enumerate(zip(growth, margin), start=1):
        rev = prev_rev * (1 + g)
        ebit = rev * m
        nopat = ebit * (1 - tax)
        reinvestment = (rev - prev_rev) / sales_to_capital
        fcff = nopat - reinvestment
        revenues.append(rev)
        ebits.append(ebit)
        fcffs.append(fcff)
        print(f"\nYear {t}:")
        print(f"  Revenue grows {g:.0%} to ${rev:.1f}")
        print(f"  Operating margin is {m:.0%}, so EBIT = ${ebit:.1f}")
        print(f"  After {tax:.0%} tax, NOPAT = ${nopat:.1f}")
        print(f"  To grow, we reinvest ${reinvestment:.1f}")
        print(f"  Free cash flow = NOPAT - reinvestment = ${fcff:.1f}")
        prev_rev = rev

    pv_fcff = sum(f / ((1 + wacc) ** t) for t, f in enumerate(fcffs, 1))
    tv = fcffs[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
    pv_tv = tv / ((1 + wacc) ** len(fcffs))
    ev = pv_fcff + pv_tv

    print(f"\nStep 2: Discount the future cash flows at {wacc:.0%}.")
    print(f"  Present value of the three years of FCFF = ${pv_fcff:.1f}")
    print(f"\nStep 3: Add a terminal value for everything after year 3.")
    print(f"  Terminal value at end of year 3 = ${tv:.1f}")
    print(f"  Discounted back to today = ${pv_tv:.1f}")
    print(f"\nStep 4: Add them up.")
    print(f"  Enterprise value today = ${ev:.1f}")
    print("\nThat is the DCF: a pile of today's dollars built from explicit")
    print("beliefs about growth, margins, reinvestment, and risk.\n")


# ---------------------------------------------------------------------------
# EXERCISE BANK (hints only; full solutions are in the learner's notebook)
# ---------------------------------------------------------------------------


class ExerciseBank:
    """
    Three practice exercises with hints.  The learner should implement
    the solution themselves; the hints point to the right part of the
    module.
    """

    @staticmethod
    def exercise_1_hint() -> str:
        return (
            "EXERCISE 1 — Reverse-engineer the WACC.\n"
            "The IPO price is $1.75T.  Using the default assumptions, find\n"
            "the WACC that makes equity_value() equal $1.75T.  Hint: write a\n"
            "small bisection loop that calls DCFModel(a).equity_value() and\n"
            "narrows the interval [0.06, 0.15] until the error is < $1B.\n"
            "Compare the answer to the base WACC of 9.0%."
        )

    @staticmethod
    def exercise_2_hint() -> str:
        return (
            "EXERCISE 2 — Build a bear case.\n"
            "Create a ValuationAssumptions where revenue grows only 30% in\n"
            "year 1, 20% in year 2, then 10% thereafter; target margin is\n"
            "15%; sales_to_capital is 1.0; WACC is 12%; terminal growth is\n"
            "2.5%.  Run DCFModel and report equity value.  Is the IPO price\n"
            "justified under these beliefs?"
        )

    @staticmethod
    def exercise_3_hint() -> str:
        return (
            "EXERCISE 3 — Probability of loss at the IPO price.\n"
            "Run MonteCarloSimulator with n_simulations=20_000 and seed=42.\n"
            "Compute the fraction of simulated equity values below $1.75T.\n"
            "Interpret the number for an Investment Committee: what does it\n"
            "say about the risk of overpaying?"
        )


# ---------------------------------------------------------------------------
# MAIN DRIVER
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the entire valuation workflow with clear console sections."""
    np.random.seed(42)

    # 1. ELI5 first, so the learner sees the mechanics before the data.
    eli5_walkthrough()

    # 2. Base-case DCF.
    print("\n" + "=" * 70)
    print("BASE-CASE DCF (default assumptions)")
    print("=" * 70)
    base = ValuationAssumptions()
    base_result = DCFModel(base).project()
    print(f"  Enterprise value : ${base_result.enterprise_value / 1e12:.3f}T")
    print(f"  Cash             : ${base.cash / 1e9:.1f}B")
    print(f"  Equity value     : ${base_result.equity_value / 1e12:.3f}T")
    print(f"  Value per share  : ${base_result.value_per_share:.2f}")
    print(f"  IPO price        : ${IPO_EQUITY_VALUE / 1e12:.2f}T")
    print(f"  Implied upside   : "
          f"{(base_result.equity_value / IPO_EQUITY_VALUE - 1) * 100:+.1f}%")

    Visualizer.plot_fcff_path(base_result)

    # 3. Scenario analysis.
    print("\n" + "=" * 70)
    print("SCENARIO ANALYSIS")
    print("=" * 70)
    scenarios = ScenarioSet().run()
    for name, r in scenarios.items():
        print(f"  {name:5s}: equity = ${r.equity_value / 1e12:.3f}T")
    Visualizer.plot_scenarios(scenarios)

    # 4. Sensitivity grid.
    print("\n" + "=" * 70)
    print("SENSITIVITY GRID (WACC vs terminal growth)")
    print("=" * 70)
    waccs, growths, grid = SensitivityGrid().run()
    Visualizer.plot_sensitivity(waccs, growths, grid)
    # Print the centre cell as a sanity check.
    mid_i = len(growths) // 2
    mid_j = len(waccs) // 2
    print(f"  Centre cell: WACC={waccs[mid_j]:.2%}, "
          f"g={growths[mid_i]:.2%} -> ${grid[mid_i, mid_j] / 1e12:.3f}T")

    # 5. Monte Carlo.
    print("\n" + "=" * 70)
    print("MONTE CARLO (10,000 simulations, seed=42)")
    print("=" * 70)
    mc = MonteCarloSimulator(seed=42, n_simulations=10_000)
    samples = mc.run()
    prob_loss = float(np.mean(samples < IPO_EQUITY_VALUE))
    print(f"  Simulations run       : {len(samples):,}")
    print(f"  Mean equity value     : ${samples.mean() / 1e12:.3f}T")
    print(f"  Median equity value   : ${np.median(samples) / 1e12:.3f}T")
    print(f"  P(equity < IPO price) : {prob_loss:.1%}")
    Visualizer.plot_monte_carlo(samples)

    # 6. Exercise hints.
    print("\n" + "=" * 70)
    print("EXERCISE HINTS")
    print("=" * 70)
    print(ExerciseBank.exercise_1_hint())
    print()
    print(ExerciseBank.exercise_2_hint())
    print()
    print(ExerciseBank.exercise_3_hint())

    print("\n" + "=" * 70)
    print("DONE — figures saved to disk.")
    print("=" * 70)


if __name__ == "__main__":
    main()
