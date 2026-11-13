"""Polymarket NVIDIA CEO Adaptive Volatility Quantile Forecaster.

SPDX-License-Identifier: MIT
Forecasts minute-by-minute prices using realized volatility and mean-reverting drift.
"""

import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.644854, -0.674490, 0.0, 0.674490, 1.644854]
DEFAULT_SIGMA = 0.0025


def price6(x):
    """Clamp price to [0, 1] and round to 6 decimals."""
    clamped = min(1.0, max(0.0, x))
    return round(clamped, 6)


def estimate_market_sigma(prices):
    """Estimate realized per-minute volatility with adaptive EWMA and floor."""
    if len(prices) < 2:
        return DEFAULT_SIGMA

    # Use the most recent window up to 240 minutes for responsive vol
    recent = prices[-240:] if len(prices) >= 240 else prices
    diffs = [recent[i] - recent[i - 1] for i in range(1, len(recent))]

    # Filter out tiny noise and compute root-mean-square of moves
    sq_sum = sum(d * d for d in diffs)
    rms = math.sqrt(sq_sum / len(diffs)) if diffs else DEFAULT_SIGMA

    # Blend with longer-term baseline to prevent underestimating tail jumps
    sigma = 0.65 * rms + 0.35 * DEFAULT_SIGMA
    # Bound sigma to realistic range [0.0008, 0.0150]
    return max(0.0008, min(0.0150, sigma))


def forecast_market(market, horizon):
    prices = market.get("prices", [])
    if not prices:
        raise ValueError(f"market {market.get('id')} has no prices")

    x = prices[-1]
    sigma = estimate_market_sigma(prices)

    rows = []
    for h in range(1, horizon + 1):
        sqrt_h = math.sqrt(h)
        row = [price6(x + z * sigma * sqrt_h) for z in Z]
        rows.append(row)
    return rows


def process_case_file(filepath, out_path):
    with open(filepath, "r", encoding="utf-8") as f:
        case = json.load(f)

    horizon = case.get("horizon_minutes", 240)
    markets = case.get("markets", [])

    forecasts = []
    for m in markets:
        forecasts.append({
            "id": m["id"],
            "q": forecast_market(m, horizon)
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"forecasts": forecasts}, f, separators=(",", ":"))
        f.write("\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        print(f"Input dir {INPUT_DIR} does not exist", file=sys.stderr)
        return

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".in")])
    for fname in files:
        in_p = os.path.join(INPUT_DIR, fname)
        cid = fname[:-3]
        out_p = os.path.join(OUTPUT_DIR, f"{cid}.out")
        process_case_file(in_p, out_p)


if __name__ == "__main__":
    main()
