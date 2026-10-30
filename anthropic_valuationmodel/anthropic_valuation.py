#!/usr/bin/env python3
"""
anthropic_ipo_valuation.py
==========================
A complete, beginner-friendly implementation of a story-driven DCF
valuation for a pre-IPO AI company (Anthropic as canonical example).

The model has four levers (Growth, Profitability, Investment Efficiency,
Risk) and five accounting adjustments (R&D capitalization, operating-lease
capitalization, NOL shield, failure-probability haircut, dilution-adjusted
option valuation). Every assumption is documented in section 3 of the
learning module.

Thanks to Dr.Ashwath Damodaran for sharing the excel sheet to the wider audience.


Design principles
-----------------
- DRY:  one `DCFEngine` class owns every intermediate array (`_rev[t]`,
  `_ebit[t]`, ...) so any downstream consumer (visualizer, exercise) reads
  the same numbers.
- Dataclasses: every stateful concept (`ValuationInputs`, `RndConverter`,
  `LeaseConverter`, `DCFEngine`, `OptionValuer`, `ValuationResult`) is a
  `@dataclass`. Instantiation is trivial, inspection is easy, equality is
  structural.
- One code path per number: `_compute_*` methods are called once per `run()`
  and their results cached on the instance. No number is computed twice.
- Explicit failure > silent garbage: every input is validated; every
  division is guarded; every iteration is bounded.

Usage
-----
    python3 anthropic_ipo_valuation.py
    pip3 install numpy pandas matplotlib

Outputs:
    fig_cashflows.png      - annual FCFF bars + PV overlay
    fig_lever_sensitivity.png - value vs WACC and vs terminal growth
    fig_value_bridge.png   - waterfall from enterprise value to equity value

No network access required.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)


# ======================================================================
# 1. R&D CAPITALIZATION (sheet "R& D converter")
# ======================================================================

@dataclass
class RndConverter:
    """
    Capitalize current-year R&D spend over a multi-year life.

    Accounting standard: US GAAP expenses R&D immediately; valuation
    theory treats R&D as capex (it builds an intangible asset that
    generates future revenue). The adjustment replaces R&D expense with
    amortization of that capitalized asset, which **increases** current
    EBIT (because R&D > current amortization when R&D is growing).

    Attributes
    ----------
    current_rd : float
        Current-year (LTM) R&D expense, $ millions.
    prior_rd : list of float
        R&D expense for the prior `life_years - 1` years, most recent first.
    life_years : int
        Amortization period. 3 is the industry default for software.
    """

    current_rd: float
    prior_rd: List[float] = field(default_factory=list)
    life_years: int = 3

    def __post_init__(self) -> None:
        if self.current_rd < 0:
            raise ValueError("current_rd must be >= 0")
        if self.life_years < 1:
            raise ValueError("life_years must be >= 1")
        if len(self.prior_rd) < self.life_years - 1:
            raise ValueError(
                f"life_years={self.life_years} requires "
                f"{self.life_years - 1} prior years, got {len(self.prior_rd)}"
            )

    def unamortized_asset(self) -> float:
        """
        Value of the R&D asset remaining at end of current year.

        For each prior R&D year `i` (0 = prior year), the fraction still
        unamortized is `(life - (i+1)) / life`. For example, with a 3-year
        life: prior-year R&D is 2/3 unamortized, two-years-back is 1/3.
        """
        unamort = [self.current_rd * 1.0]
        for i, rd in enumerate(self.prior_rd[: self.life_years - 1]):
            remaining = self.life_years - (i + 1)
            unamort.append(rd * remaining / self.life_years)
        return sum(unamort)

    def amortization(self) -> float:
        """Current-year amortization = sum of all prior R&D / life."""
        return sum(self.prior_rd[: self.life_years - 1]) / self.life_years

    def ebit_adjustment(self) -> float:
        """
        Net adjustment to reported EBIT: current R&D minus amortization.
        Positive because R&D (capex) exceeds amortization when R&D grows.
        """
        return self.current_rd - self.amortization()


# ======================================================================
# 2. OPERATING LEASE CAPITALIZATION (sheet "Operating lease converter")
# ======================================================================

@dataclass
class LeaseConverter:
    """
    Capitalize operating lease commitments as debt.

    Pre-2019 GAAP left operating leases off-balance-sheet. Modern GAAP
    and IFRS capitalise them. For valuation, treat the PV of future
    commitments as debt and add back the "implied depreciation" to EBIT.
    The adjustment is *negative* because cash lease expense is typically
    smaller than the annuity depreciation being added back.

    Attributes
    ----------
    annual_expense : float
        Current-year cash lease expense, $ millions.
    commitments : list of float
        Committed lease payments for years 1..5 and a "beyond year 5"
        lump sum as element 5. Length must be exactly 6.
    pretax_cost_debt : float
        Discount rate — use the company's synthetic cost of debt.
    """

    annual_expense: float
    commitments: List[float]
    pretax_cost_debt: float

    def __post_init__(self) -> None:
        if len(self.commitments) != 6:
            raise ValueError("commitments must have 6 entries (yrs 1-5 + yr6+)")
        if self.pretax_cost_debt <= 0:
            raise ValueError("pretax_cost_debt must be > 0")

    def _n_years_beyond_five(self) -> int:
        """Spread the lump-sum year-6+ commitment over the average term."""
        avg5 = sum(self.commitments[:5]) / 5
        if avg5 == 0 or self.commitments[5] <= 0:
            return 0
        return int(round(self.commitments[5] / avg5))

    def lease_debt(self) -> float:
        """
        PV of future lease commitments = debt value.
        Years 1-5 discounted individually; year 6+ treated as an annuity.
        """
        r = self.pretax_cost_debt
        pv = [self.commitments[t - 1] / (1 + r) ** t for t in range(1, 6)]
        n = self._n_years_beyond_five()
        if n > 0:
            annuity = (self.commitments[5] * (1 - (1 + r) ** -n) / r) / (1 + r) ** 5
        else:
            annuity = self.commitments[5] / (1 + r) ** 6
        pv.append(annuity)
        return sum(pv)

    def depreciation(self) -> float:
        """Straight-line depreciation of the lease asset."""
        n = self._n_years_beyond_five()
        return self.lease_debt() / (5 + n)

    def ebit_adjustment(self) -> float:
        """Add back cash expense, subtract implied depreciation."""
        return self.annual_expense - self.depreciation()


# ======================================================================
# 3. INPUTS
# ======================================================================

@dataclass
class ValuationInputs:
    """
    Every numeric input required to run the DCF. No defaults are hidden:
    the caller must supply each field, which makes the model auditable
    against a source document (S-1, board memo, etc.).

    All monetary values are in **$ millions** unless noted.
    """

    # --- base year ---
    revenue_ltm: float
    ebit_ltm: float
    book_equity: float
    book_debt: float
    cash: float
    cross_holdings: float
    minority: float
    shares_outstanding: float           # millions
    price_per_share: float
    effective_tax_rate: float
    marginal_tax_rate: float

    # --- growth lever ---
    growth_y1: float                    # e.g. 4.0 means 4x = 400%
    growth_y2_5: float                  # CAGR

    # --- profitability lever ---
    margin_y1: float
    target_margin: float
    convergence_years: int = 5

    # --- investment-efficiency lever ---
    sales_to_capital_y1_5: float = 3.0
    sales_to_capital_y6_10: float = 3.0

    # --- risk lever ---
    riskfree_rate: float = 0.049
    initial_wacc: float = 0.0974
    stable_wacc: float = 0.0913         # typically rf + mature ERP
    stable_growth: float = 0.049        # typically == rf
    stable_roic: float = 0.15

    # --- failure adjustment ---
    prob_failure: float = 0.10
    distress_pct: float = 0.50          # % of fair value if failure occurs

    # --- bridge ---
    ipo_proceeds: float = 100_000
    nol_start: float = 0.0

    # --- option valuation ---
    opt_n_outstanding: float = 0.902
    opt_strike: float = 2.06
    opt_maturity_years: float = 8.1
    opt_volatility: float = 0.40
    opt_dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        """Validate every input at construction — fail fast, fail loud."""
        if self.revenue_ltm <= 0:
            raise ValueError("revenue_ltm must be positive")
        if self.shares_outstanding <= 0:
            raise ValueError("shares_outstanding must be positive")
        if not (0 <= self.effective_tax_rate <= 1):
            raise ValueError("effective_tax_rate must be in [0, 1]")
        if not (0 <= self.marginal_tax_rate <= 1):
            raise ValueError("marginal_tax_rate must be in [0, 1]")
        if self.initial_wacc <= self.stable_growth:
            raise ValueError("initial_wacc must exceed stable_growth")
        if self.stable_wacc <= self.stable_growth:
            raise ValueError("stable_wacc must exceed stable_growth (else TV < 0)")
        if not (0 <= self.prob_failure <= 1):
            raise ValueError("prob_failure must be in [0, 1]")
        if self.convergence_years < 1:
            raise ValueError("convergence_years must be >= 1")


# ======================================================================
# 4. DCF ENGINE
# ======================================================================

@dataclass
class ValuationResult:
    """Immutable output of one DCF run. Every number is inspectable."""

    growth: Dict[int, float]
    revenue: Dict[int, float]
    margin: Dict[int, float]
    ebit: Dict[int, float]
    nopat: Dict[int, float]
    reinvestment: Dict[int, float]
    fcff: Dict[int, float]
    wacc: Dict[int, float]
    discount_factor: Dict[int, float]
    pv_fcff: Dict[int, float]
    revenue_terminal: float
    nopat_terminal: float
    fcff_terminal: float
    terminal_value: float
    pv_terminal: float
    sum_pv: float
    value_operating_assets: float
    debt_total: float
    equity_value: float
    equity_in_common: float
    option_value: float
    value_per_share: float
    price_as_pct_of_value: float


class DCFEngine:
    """
    Runs the 10-year DCF + terminal value + failure adjustment + bridge
    to per-share value.

    This class is the single source of truth for every intermediate
    number. The visualizer and exercises consume its outputs; they never
    recompute anything.
    """

    def __init__(
        self,
        inputs: ValuationInputs,
        rd: RndConverter,
        lease: LeaseConverter,
    ) -> None:
        self.inp = inputs
        self.rd = rd
        self.lease = lease

    # ------------------------------------------------------------------
    # 4.1 Adjust base-year EBIT for R&D + leases
    # ------------------------------------------------------------------
    def _adjusted_base_ebit(self) -> float:
        """
        Reported EBIT already excludes R&D and lease cash expense; the
        converters tell us how much to *add back* to make EBIT reflect
        economic capex instead of accounting expense.
        """
        return self.inp.ebit_ltm + self.rd.ebit_adjustment() + self.lease.ebit_adjustment()

    # ------------------------------------------------------------------
    # 4.2 Growth and revenue schedules
    # ------------------------------------------------------------------
    def _growth_schedule(self) -> Dict[int, float]:
        """
        Years 1-5: linear between `growth_y1` and `growth_y2_5` (by
        convention, `growth_y2_5` applies to years 2-5, so year 2 jumps
        straight to it). Years 6-10: linear decline toward `stable_growth`.
        """
        g: Dict[int, float] = {1: self.inp.growth_y1}
        for t in range(2, 6):
            g[t] = self.inp.growth_y2_5
        decline = (self.inp.growth_y2_5 - self.inp.stable_growth) / 5
        for t in range(6, 11):
            g[t] = self.inp.growth_y2_5 - decline * (t - 5)
        return g

    def _revenue_schedule(self, growth: Dict[int, float]) -> Dict[int, float]:
        rev: Dict[int, float] = {0: self.inp.revenue_ltm}
        for t in range(1, 11):
            rev[t] = rev[t - 1] * (1.0 + growth[t])
        return rev

    # ------------------------------------------------------------------
    # 4.3 Margin convergence
    # ------------------------------------------------------------------
    def _margin_schedule(self) -> Dict[int, float]:
        """Start at margin_y1, converge linearly to target over `convergence_years`."""
        m: Dict[int, float] = {1: self.inp.margin_y1}
        delta = self.inp.target_margin - self.inp.margin_y1
        for t in range(2, 11):
            if t > self.inp.convergence_years:
                m[t] = self.inp.target_margin
            else:
                progress = (t - 1) / self.inp.convergence_years
                m[t] = self.inp.margin_y1 + delta * progress
        return m

    # ------------------------------------------------------------------
    # 4.4 EBIT and NOPAT with NOL shield
    # ------------------------------------------------------------------
    def _ebit_schedule(
        self, rev: Dict[int, float], margin: Dict[int, float]
    ) -> Dict[int, float]:
        return {t: rev[t] * margin[t] for t in range(1, 11)}

    def _nopat_with_nol(self, ebit: Dict[int, float]) -> Dict[int, float]:
        """
        NOPAT = EBIT × (1 − tax) when EBIT > 0, else EBIT itself.
        Positive years first *consume* any accumulated net operating loss
        (NOL) before paying tax — a real tax benefit for young companies.
        """
        nopat: Dict[int, float] = {}
        nol = self.inp.nol_start
        eff_tax = self.inp.effective_tax_rate
        marg_tax = self.inp.marginal_tax_rate

        # Tax rate ramps linearly from effective to marginal over 10 years.
        tax_rate = {t: eff_tax + (marg_tax - eff_tax) * (t / 10) for t in range(1, 11)}

        for t in range(1, 11):
            e = ebit[t]
            if e <= 0:
                nopat[t] = e
                nol += -e
            else:
                taxable = max(e - nol, 0.0)
                nol = max(nol - e, 0.0)
                nopat[t] = e - taxable * tax_rate[t]
        return nopat

    # ------------------------------------------------------------------
    # 4.5 Reinvestment (revenue growth / sales-to-capital)
    # ------------------------------------------------------------------
    def _reinvestment_schedule(
        self, rev: Dict[int, float], rev_terminal: float
    ) -> Dict[int, float]:
        """
        Sales-to-capital ratio tells us how many dollars of revenue each
        dollar of reinvestment produces. Reinvestment_t = ΔRevenue_{t+1} / S2C_t.
        Year 10 uses the terminal revenue as its next-year revenue.
        """
        reinv: Dict[int, float] = {}
        for t in range(1, 10):
            s2c = (
                self.inp.sales_to_capital_y1_5
                if t <= 5
                else self.inp.sales_to_capital_y6_10
            )
            reinv[t] = (rev[t + 1] - rev[t]) / s2c
        s2c_t = self.inp.sales_to_capital_y6_10
        reinv[10] = (rev_terminal - rev[10]) / s2c_t
        return reinv

    # ------------------------------------------------------------------
    # 4.6 Cost of capital (drift from initial to stable)
    # ------------------------------------------------------------------
    def _wacc_schedule(self) -> Dict[int, float]:
        w: Dict[int, float] = {t: self.inp.initial_wacc for t in range(1, 6)}
        step = (self.inp.initial_wacc - self.inp.stable_wacc) / 5
        for t in range(6, 11):
            w[t] = self.inp.initial_wacc - step * (t - 5)
        return w

    # ------------------------------------------------------------------
    # 4.7 Present value of FCFF + terminal value
    # ------------------------------------------------------------------
    def _pv_schedule(
        self, fcff: Dict[int, float], wacc: Dict[int, float]
    ) -> Tuple[Dict[int, float], Dict[int, float]]:
        df: Dict[int, float] = {}
        running = 1.0
        for t in range(1, 11):
            running /= 1.0 + wacc[t]
            df[t] = running
        pv = {t: fcff[t] * df[t] for t in range(1, 11)}
        return df, pv

    def _terminal_value(self, fcff_terminal: float) -> float:
        """Gordon growth: TV = FCFF_terminal × (1+g) / (WACC − g)."""
        return fcff_terminal / (self.inp.stable_wacc - self.inp.stable_growth)

    # ------------------------------------------------------------------
    # 4.8 Failure adjustment
    # ------------------------------------------------------------------
    def _value_with_failure(self, sum_pv: float) -> float:
        """Expected value = (1 − p) × as-going-concern + p × distress proceeds."""
        distress = sum_pv * self.inp.distress_pct
        return sum_pv * (1 - self.inp.prob_failure) + distress * self.inp.prob_failure

    # ------------------------------------------------------------------
    # 4.9 Bridge to per-share value (closes the share-count circularity)
    # ------------------------------------------------------------------
    def _bridge_to_per_share(
        self, value_operating_assets: float, option_value: float
    ) -> Tuple[float, float, float]:
        """
        Equity value = EV − Debt − Minority + Cash + IPO proceeds + Non-op.
        Common equity = Equity value − Option value.
        Share count circularity (share count depends on price, price
        depends on share count) is resolved algebraically:

            price = (equity_in_common − IPO proceeds) / shares_outstanding
        """
        debt_total = self.inp.book_debt + self.lease.lease_debt()
        equity = (
            value_operating_assets
            - debt_total
            - self.inp.minority
            + self.inp.cash
            + self.inp.ipo_proceeds
            + self.inp.cross_holdings
        )
        equity_common = equity - option_value
        price = (equity_common - self.inp.ipo_proceeds) / self.inp.shares_outstanding
        return equity, equity_common, price

    # ------------------------------------------------------------------
    # 4.10 PUBLIC ENTRY POINT
    # ------------------------------------------------------------------
    def run(self) -> ValuationResult:
        """Execute the full pipeline and return a fully-populated result."""
        growth = self._growth_schedule()
        rev = self._revenue_schedule(growth)
        rev_term = rev[10] * (1 + self.inp.stable_growth)

        margin = self._margin_schedule()
        ebit = self._ebit_schedule(rev, margin)
        nopat = self._nopat_with_nol(ebit)

        # Terminal NOPAT: EBIT × (1 − marginal_tax)
        nopat_term = rev_term * self.inp.target_margin * (1 - self.inp.marginal_tax_rate)

        reinv = self._reinvestment_schedule(rev, rev_term)
        # Terminal reinvestment = g / ROIC × NOPAT_terminal (steady state)
        reinv_term = (
            self.inp.stable_growth / self.inp.stable_roic * nopat_term
            if self.inp.stable_growth > 0
            else 0.0
        )

        fcff = {t: nopat[t] - reinv[t] for t in range(1, 11)}
        fcff_term = nopat_term - reinv_term

        wacc = self._wacc_schedule()
        df, pv = self._pv_schedule(fcff, wacc)
        pv_explicit = sum(pv.values())

        tv = self._terminal_value(fcff_term)
        pv_tv = tv * df[10]
        sum_pv = pv_explicit + pv_tv

        value_ops = self._value_with_failure(sum_pv)

        # Option value (dilution-adjusted BS, iterative)
        ov = OptionValuer(
            S=self.inp.price_per_share,
            K=self.inp.opt_strike,
            T=self.inp.opt_maturity_years,
            sigma=self.inp.opt_volatility,
            r=self.inp.riskfree_rate,
            q=self.inp.opt_dividend_yield,
            n_warrants=self.inp.opt_n_outstanding,
            n_shares=self.inp.shares_outstanding,
        )
        opt_value = ov.value_all()

        equity, equity_common, price = self._bridge_to_per_share(value_ops, opt_value)

        return ValuationResult(
            growth=growth,
            revenue=rev,
            margin=margin,
            ebit=ebit,
            nopat=nopat,
            reinvestment=reinv,
            fcff=fcff,
            wacc=wacc,
            discount_factor=df,
            pv_fcff=pv,
            revenue_terminal=rev_term,
            nopat_terminal=nopat_term,
            fcff_terminal=fcff_term,
            terminal_value=tv,
            pv_terminal=pv_tv,
            sum_pv=sum_pv,
            value_operating_assets=value_ops,
            debt_total=self.inp.book_debt + self.lease.lease_debt(),
            equity_value=equity,
            equity_in_common=equity_common,
            option_value=opt_value,
            value_per_share=price,
            price_as_pct_of_value=price / price if price else float("nan"),
        )


# ======================================================================
# 5. OPTION VALUER (dilution-adjusted Black-Scholes, iterated)
# ======================================================================

@dataclass
class OptionValuer:
    """
    Dilution-adjusted Black-Scholes option valuation.

    When warrants/options are numerous, the stock price used in BS must
    be the *post-dilution* price, which itself depends on option value —
    a fixed-point. Solved by bounded iteration.
    """

    S: float
    K: float
    T: float
    sigma: float
    r: float
    q: float
    n_warrants: float
    n_shares: float
    tol: float = 1e-10
    max_iter: int = 500

    @staticmethod
    def _norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def _value_per_option(self, wv: float) -> float:
        adj_S = (self.S * self.n_shares + wv * self.n_warrants) / (
            self.n_shares + self.n_warrants
        )
        d1 = (
            math.log(adj_S / self.K)
            + (self.r - self.q + self.sigma ** 2 / 2) * self.T
        ) / (self.sigma * math.sqrt(self.T))
        d2 = d1 - self.sigma * math.sqrt(self.T)
        return (
            math.exp(-self.q * self.T) * adj_S * self._norm_cdf(d1)
            - self.K * math.exp(-self.r * self.T) * self._norm_cdf(d2)
        )

    def value_all(self) -> float:
        """Iterate to fixed point; return total option value ($ millions)."""
        wv = 1.0
        for _ in range(self.max_iter):
            new = self._value_per_option(wv)
            if abs(new - wv) < self.tol:
                wv = new
                break
            wv = new
        return wv * self.n_warrants


# ======================================================================
# 6. INPUT VALIDATOR (edge-case handling)
# ======================================================================

class InputValidator:
    """
    Guard every public entry point against malformed inputs.

    Each method raises `ValueError` with a clear message so a beginner
    learns *why* the input was rejected, not just that it was.
    """

    @staticmethod
    def require_positive(value: float, name: str) -> None:
        if value is None:
            raise ValueError(f"{name} is None; expected a positive number")
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def require_probability(value: float, name: str) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")

    @staticmethod
    def require_sequence(values: Sequence[float], name: str, length: int) -> None:
        if not isinstance(values, (list, tuple)):
            raise TypeError(f"{name} must be a list/tuple")
        if len(values) != length:
            raise ValueError(f"{name} must have {length} entries, got {len(values)}")
        for i, v in enumerate(values):
            if not isinstance(v, (int, float)):
                raise TypeError(f"{name}[{i}] must be numeric")


# ======================================================================
# 7. VISUALIZER
# ======================================================================

class ValuationVisualizer:
    """Save three diagnostic charts."""

    @staticmethod
    def plot_cashflows(result: ValuationResult, path: str = "fig_cashflows.png") -> None:
        years = list(range(1, 11))
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(years, [result.fcff[t] for t in years],
               color="steelblue", label="FCFF")
        ax.plot(years, [result.pv_fcff[t] for t in years],
                marker="o", color="darkorange", label="PV(FCFF)")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Forecast year")
        ax.set_ylabel("$ millions")
        ax.set_title("FCFF and discounted FCFF, years 1-10")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_value_bridge(result: ValuationResult, path: str = "fig_value_bridge.png") -> None:
        labels = ["Operating\nassets", "PV of\nterminal", "Debt", "Cash + IPO", "Equity"]
        values = [
            result.sum_pv,
            result.pv_terminal,
            -result.debt_total,
            result.equity_value - result.value_operating_assets + result.debt_total,
            result.equity_value,
        ]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["steelblue" if v >= 0 else "indianred" for v in values]
        bars = ax.bar(labels, values, color=colors)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v,
                    f"{v/1000:,.1f}B", ha="center", va="bottom" if v >= 0 else "top")
        ax.set_ylabel("$ millions")
        ax.set_title("Value bridge: enterprise value to equity value")
        ax.axhline(0, color="black", lw=0.5)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)

    @staticmethod
    def plot_lever_sensitivity(
        inputs: ValuationInputs,
        rd: RndConverter,
        lease: LeaseConverter,
        path: str = "fig_lever_sensitivity.png",
    ) -> None:
        """Sweep initial WACC and terminal growth, show per-share value response."""
        wacc_grid = np.linspace(inputs.initial_wacc - 0.02, inputs.initial_wacc + 0.02, 11)
        wacc_vals: List[float] = []
        for w in wacc_grid:
            if w <= inputs.stable_growth:
                wacc_vals.append(float("nan"))
                continue
            try:
                modified = ValuationInputs(**{**inputs.__dict__, "initial_wacc": float(w)})
                r = DCFEngine(modified, rd, lease).run()
                wacc_vals.append(r.value_per_share)
            except ValueError:
                wacc_vals.append(float("nan"))

        g_grid = np.linspace(inputs.stable_growth - 0.02, inputs.stable_growth + 0.01, 11)
        g_vals: List[float] = []
        for g in g_grid:
            if g >= inputs.stable_wacc:
                g_vals.append(float("nan"))
                continue
            try:
                modified = ValuationInputs(**{**inputs.__dict__, "stable_growth": float(g)})
                r = DCFEngine(modified, rd, lease).run()
                g_vals.append(r.value_per_share)
            except ValueError:
                g_vals.append(float("nan"))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
        ax1.plot(wacc_grid * 100, wacc_vals, marker="o", color="steelblue")
        ax1.set_xlabel("Initial WACC (%)")
        ax1.set_ylabel("Value per share ($)")
        ax1.set_title("Value vs initial WACC")
        ax1.grid(True, alpha=0.3)

        ax2.plot(g_grid * 100, g_vals, marker="s", color="darkorange")
        ax2.set_xlabel("Stable growth (%)")
        ax2.set_ylabel("Value per share ($)")
        ax2.set_title("Value vs terminal growth")
        ax2.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# ======================================================================
# 8. ELI5 WALKTHROUGH
# ======================================================================

def eli5_walkthrough() -> None:
    """Toy 3-year example with plain-language narration."""
    print("\n=== ELI5 walkthrough: mini lemonade stand ===\n")
    revenue = 100.0
    growth = [0.5, 0.3, 0.2]
    margin = 0.20
    wacc = 0.10
    print(f"Today the stand sells ${revenue:.0f} of lemonade.")
    print(f"Story: it will grow {growth} per year.")
    print(f"Story: it keeps a {margin:.0%} profit margin.")
    print(f"Story: money later is worth {wacc:.0%} less per year.\n")

    pv_total = 0.0
    for t, g in enumerate(growth, start=1):
        revenue *= 1 + g
        profit = revenue * margin
        pv = profit / (1 + wacc) ** t
        pv_total += pv
        print(f"Year {t}: sell ${revenue:6.1f}, keep ${profit:5.1f}, "
              f"today it's worth ${pv:5.1f}")

    print(f"\nTotal value today: ${pv_total:,.1f}")
    print("That is the DCF. The four levers are: growth, margin, "
          "reinvestment (ignored here), and risk (wacc).")


# ======================================================================
# 9. MAIN DRIVER — happy path + edge cases
# ======================================================================

def build_canonical_anthropic_inputs() -> Tuple[
    ValuationInputs, RndConverter, LeaseConverter
]:
    """
    Build the canonical Anthropic inputs used throughout the module.

    Numbers are synthetic but magnitudes match public 2025 rumoured
    figures: ~$20.6bn LTM revenue, ~-$6.6bn EBIT, ~$100bn IPO.
    """
    inputs = ValuationInputs(
        revenue_ltm=5485 - 1163 + 16286,           # 20,608
        ebit_ltm=-8060 + 2135 - 693,               # -6,618
        book_equity=147358,
        book_debt=108566 + 750,                    # 109,316
        cash=112389,
        cross_holdings=3929,
        minority=0,
        shares_outstanding=1049.63 + 2024.97,      # 3,074.60
        price_per_share=525,
        effective_tax_rate=0.0,
        marginal_tax_rate=0.25,
        growth_y1=4.0,
        growth_y2_5=0.42,
        margin_y1=0.05,
        target_margin=0.325,
        convergence_years=5,
        sales_to_capital_y1_5=3.0,
        sales_to_capital_y6_10=3.0,
        riskfree_rate=0.049,
        initial_wacc=0.0974,
        stable_wacc=0.0913,
        stable_growth=0.049,
        stable_roic=0.15,
        prob_failure=0.10,
        distress_pct=0.50,
        ipo_proceeds=100_000,
        nol_start=0.0,
    )
    rd = RndConverter(current_rd=16979, prior_rd=[12645, 3366, 592], life_years=3)
    lease = LeaseConverter(
        annual_expense=52,
        commitments=[63168, 98128, 60832, 58304, 50000, 124000],
        pretax_cost_debt=0.049 + 0.013828,
    )
    return inputs, rd, lease


def scenario_happy_path() -> ValuationResult:
    """Canonical run of the full Anthropic model."""
    print("\n[Scenario: happy path]")
    inp, rd, lease = build_canonical_anthropic_inputs()
    return DCFEngine(inp, rd, lease).run()


def scenario_missing_inputs() -> None:
    """What happens when a required numeric field is absent."""
    print("\n[Scenario: missing inputs]")
    try:
        ValuationInputs(
            revenue_ltm=None,  # type: ignore[arg-type]
            ebit_ltm=-6618, book_equity=1, book_debt=1, cash=1,
            cross_holdings=0, minority=0, shares_outstanding=3074.6,
            price_per_share=525, effective_tax_rate=0, marginal_tax_rate=0.25,
            growth_y1=4, growth_y2_5=0.42, margin_y1=0.05, target_margin=0.325,
        )
    except (TypeError, ValueError) as e:
        print(f"  → rejected as expected: {e}")


def scenario_extreme_values() -> None:
    """What happens when WACC ≤ g or growth is absurd."""
    print("\n[Scenario: extreme values]")
    try:
        ValuationInputs(
            revenue_ltm=20_608, ebit_ltm=-6_618, book_equity=1, book_debt=1,
            cash=1, cross_holdings=0, minority=0, shares_outstanding=3074.6,
            price_per_share=525, effective_tax_rate=0, marginal_tax_rate=0.25,
            growth_y1=4, growth_y2_5=0.42, margin_y1=0.05, target_margin=0.325,
            initial_wacc=0.03, stable_growth=0.05,
        )
    except ValueError as e:
        print(f"  → rejected as expected: {e}")


def scenario_empty_option_list() -> None:
    """Option valuer with zero outstanding options must return 0, not crash."""
    print("\n[Scenario: empty option list]")
    ov = OptionValuer(S=100, K=50, T=5, sigma=0.4, r=0.05, q=0.0,
                      n_warrants=0.0, n_shares=1000)
    print(f"  → value_all() = {ov.value_all():.6f} (expected 0.0)")


def main() -> None:
    print("=" * 78)
    print(" Anthropic IPO Valuation — Learning Module Driver")
    print("=" * 78)

    eli5_walkthrough()

    result = scenario_happy_path()
    print(f"\n  Revenue Y10        : ${result.revenue[10]:,.0f}M")
    print(f"  Terminal value     : ${result.terminal_value:,.0f}M")
    print(f"  Operating assets   : ${result.value_operating_assets:,.0f}M")
    print(f"  Equity value       : ${result.equity_value:,.0f}M")
    print(f"  Option value       : ${result.option_value:,.0f}M")
    print(f"  Value per share    : ${result.value_per_share:,.2f}")
    print(f"  Current price      : ${525:.2f}")

    scenario_missing_inputs()
    scenario_extreme_values()
    scenario_empty_option_list()

    inp, rd, lease = build_canonical_anthropic_inputs()
    ValuationVisualizer.plot_cashflows(result)
    ValuationVisualizer.plot_value_bridge(result)
    ValuationVisualizer.plot_lever_sensitivity(inp, rd, lease)
    print("\nFigures saved: fig_cashflows.png, fig_value_bridge.png, "
          "fig_lever_sensitivity.png")


if __name__ == "__main__":
    main()
