#!/usr/bin/env python3
"""
IMC Prosperity Round 1 data analyser.

Put this file in the same folder as:
  prices_round_1_day_-2.csv / prices_round_1_day_-2
  prices_round_1_day_-1.csv / prices_round_1_day_-1
  prices_round_1_day_0.csv  / prices_round_1_day_0
  trades_round_1_day_-2.csv / trades_round_1_day_-2
  trades_round_1_day_-1.csv / trades_round_1_day_-1
  trades_round_1_day_0.csv  / trades_round_1_day_0

Run:
  python analyse_round1.py

Outputs are written to:
  analysis_outputs/
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DAYS = [-2, -1, 0]
OUT = Path("analysis_outputs")
OUT.mkdir(exist_ok=True)


def find_file(prefix: str, day: int) -> Path:
    candidates = [
        Path(f"{prefix}_round_1_day_{day}.csv"),
        Path(f"{prefix}_round_1_day_{day}"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Missing {prefix}_round_1_day_{day}(.csv)")


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if c not in {"product", "symbol", "buyer", "seller", "currency"}:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_prices() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = find_file("prices", d)
        df = pd.read_csv(p, sep=";")
        df.columns = df.columns.str.strip()
        df = clean_numeric(df)
        frames.append(df)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)

    # Depth/spread features
    prices["best_bid"] = prices["bid_price_1"]
    prices["best_ask"] = prices["ask_price_1"]
    prices["spread"] = prices["best_ask"] - prices["best_bid"]

    bid_vol_cols = [f"bid_volume_{i}" for i in range(1, 4)]
    ask_vol_cols = [f"ask_volume_{i}" for i in range(1, 4)]

    prices["bid_depth"] = prices[bid_vol_cols].abs().sum(axis=1, skipna=True)
    prices["ask_depth"] = prices[ask_vol_cols].abs().sum(axis=1, skipna=True)
    prices["depth_imbalance"] = (
        (prices["bid_depth"] - prices["ask_depth"])
        / (prices["bid_depth"] + prices["ask_depth"]).replace(0, np.nan)
    )

    prices["book_mid"] = (prices["best_bid"] + prices["best_ask"]) / 2
    prices["mid_return"] = prices.groupby(["day", "product"])["mid_price"].diff()

    return prices


def load_trades() -> pd.DataFrame:
    frames = []
    for d in DAYS:
        p = find_file("trades", d)
        df = pd.read_csv(p, sep=";")
        df.columns = df.columns.str.strip()
        df = clean_numeric(df)
        df["day"] = d
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)
    trades = trades.rename(columns={"symbol": "product"})
    trades = trades.sort_values(["day", "timestamp", "product"]).reset_index(drop=True)
    trades["notional"] = trades["price"] * trades["quantity"]
    return trades


def save_basic_summaries(prices: pd.DataFrame, trades: pd.DataFrame) -> None:
    price_summary = (
        prices.groupby(["day", "product"])
        .agg(
            rows=("timestamp", "count"),
            mid_mean=("mid_price", "mean"),
            mid_std=("mid_price", "std"),
            mid_min=("mid_price", "min"),
            mid_max=("mid_price", "max"),
            spread_mean=("spread", "mean"),
            spread_min=("spread", "min"),
            spread_max=("spread", "max"),
            bid_depth_mean=("bid_depth", "mean"),
            ask_depth_mean=("ask_depth", "mean"),
            imbalance_mean=("depth_imbalance", "mean"),
            imbalance_std=("depth_imbalance", "std"),
        )
        .reset_index()
    )

    trade_summary = (
        trades.groupby(["day", "product"])
        .agg(
            trades=("timestamp", "count"),
            total_quantity=("quantity", "sum"),
            avg_quantity=("quantity", "mean"),
            price_mean=("price", "mean"),
            price_std=("price", "std"),
            price_min=("price", "min"),
            price_max=("price", "max"),
            notional=("notional", "sum"),
        )
        .reset_index()
    )

    trade_summary["vwap"] = trade_summary["notional"] / trade_summary["total_quantity"]

    overall = (
        prices.groupby("product")
        .agg(
            rows=("timestamp", "count"),
            mid_mean=("mid_price", "mean"),
            mid_std=("mid_price", "std"),
            mid_min=("mid_price", "min"),
            mid_max=("mid_price", "max"),
            spread_mean=("spread", "mean"),
            bid_depth_mean=("bid_depth", "mean"),
            ask_depth_mean=("ask_depth", "mean"),
        )
        .reset_index()
    )

    price_summary.to_csv(OUT / "price_summary_by_day_product.csv", index=False)
    trade_summary.to_csv(OUT / "trade_summary_by_day_product.csv", index=False)
    overall.to_csv(OUT / "overall_product_summary.csv", index=False)


def attach_trade_context(prices: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    # Join each trade to the nearest available book row at the same day/product/timestamp.
    # Usually timestamps match exactly in Prosperity data.
    ctx_cols = [
        "day", "timestamp", "product",
        "mid_price", "best_bid", "best_ask", "spread",
        "bid_depth", "ask_depth", "depth_imbalance",
    ]

    out = trades.merge(
        prices[ctx_cols],
        on=["day", "timestamp", "product"],
        how="left",
    )

    # Edge relative to observed mid. Positive means trade happened above mid.
    # Direction is unknown because public trades do not always reveal whether
    # the trade was buyer-initiated or seller-initiated.
    out["trade_minus_mid"] = out["price"] - out["mid_price"]
    out["abs_trade_edge_vs_mid"] = out["trade_minus_mid"].abs()

    out.to_csv(OUT / "trades_with_book_context.csv", index=False)
    return out


def analyse_predictability(prices: pd.DataFrame) -> None:
    rows = []
    horizons = [1, 2, 5, 10]

    for (day, product), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp").copy()

        for h in horizons:
            future = g["mid_price"].shift(-h) - g["mid_price"]
            rows.append({
                "day": day,
                "product": product,
                "horizon_steps": h,
                "corr_imbalance_future_mid_change": g["depth_imbalance"].corr(future),
                "corr_spread_future_abs_change": g["spread"].corr(future.abs()),
                "avg_future_mid_change": future.mean(),
                "std_future_mid_change": future.std(),
            })

    pd.DataFrame(rows).to_csv(OUT / "simple_predictability_checks.csv", index=False)


def plot_mid_and_spread(prices: pd.DataFrame) -> None:
    for product, g in prices.groupby("product"):
        g = g.sort_values(["day", "timestamp"]).copy()
        g["global_t"] = g["timestamp"] + (g["day"] - min(DAYS)) * 1_000_000

        plt.figure(figsize=(14, 5))
        for d, gd in g.groupby("day"):
            plt.plot(gd["global_t"], gd["mid_price"], linewidth=1, label=f"day {d}")
        plt.title(f"{product}: mid price over time")
        plt.xlabel("global timestamp")
        plt.ylabel("mid price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{safe_name(product)}_mid_price.png", dpi=160)
        plt.close()

        plt.figure(figsize=(14, 4))
        for d, gd in g.groupby("day"):
            plt.plot(gd["global_t"], gd["spread"], linewidth=1, label=f"day {d}")
        plt.title(f"{product}: spread over time")
        plt.xlabel("global timestamp")
        plt.ylabel("best ask - best bid")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{safe_name(product)}_spread.png", dpi=160)
        plt.close()


def plot_orderbook_cloud(prices: pd.DataFrame, trades_ctx: pd.DataFrame) -> None:
    for (day, product), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp")

        plt.figure(figsize=(15, 6))

        # Plot top 3 bid/ask levels. Marker size reflects volume.
        for i in range(1, 4):
            bp, bv = f"bid_price_{i}", f"bid_volume_{i}"
            ap, av = f"ask_price_{i}", f"ask_volume_{i}"

            if bp in g:
                plt.scatter(
                    g["timestamp"], g[bp],
                    s=g[bv].abs().fillna(0) * 4,
                    alpha=0.35,
                    label=f"bid {i}" if i == 1 else None,
                )
            if ap in g:
                plt.scatter(
                    g["timestamp"], g[ap],
                    s=g[av].abs().fillna(0) * 4,
                    alpha=0.35,
                    label=f"ask {i}" if i == 1 else None,
                )

        plt.plot(g["timestamp"], g["mid_price"], linewidth=1, label="mid")

        tg = trades_ctx[(trades_ctx["day"] == day) & (trades_ctx["product"] == product)]
        if not tg.empty:
            plt.scatter(
                tg["timestamp"], tg["price"],
                s=tg["quantity"].fillna(1) * 12,
                marker="x",
                label="trades",
            )

        plt.title(f"{product}, day {day}: order book + trades")
        plt.xlabel("timestamp")
        plt.ylabel("price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{safe_name(product)}_day_{day}_orderbook.png", dpi=170)
        plt.close()


def plot_normalized_orderbook(prices: pd.DataFrame, trades_ctx: pd.DataFrame) -> None:
    for (day, product), g in prices.groupby(["day", "product"]):
        g = g.sort_values("timestamp").copy()

        plt.figure(figsize=(15, 6))

        for i in range(1, 4):
            bp, bv = f"bid_price_{i}", f"bid_volume_{i}"
            ap, av = f"ask_price_{i}", f"ask_volume_{i}"

            if bp in g:
                plt.scatter(
                    g["timestamp"], g[bp] - g["mid_price"],
                    s=g[bv].abs().fillna(0) * 4,
                    alpha=0.35,
                    label=f"bid {i}" if i == 1 else None,
                )
            if ap in g:
                plt.scatter(
                    g["timestamp"], g[ap] - g["mid_price"],
                    s=g[av].abs().fillna(0) * 4,
                    alpha=0.35,
                    label=f"ask {i}" if i == 1 else None,
                )

        tg = trades_ctx[(trades_ctx["day"] == day) & (trades_ctx["product"] == product)]
        if not tg.empty:
            plt.scatter(
                tg["timestamp"], tg["price"] - tg["mid_price"],
                s=tg["quantity"].fillna(1) * 12,
                marker="x",
                label="trades - mid",
            )

        plt.axhline(0, linewidth=1)
        plt.title(f"{product}, day {day}: normalized order book around mid")
        plt.xlabel("timestamp")
        plt.ylabel("price - mid_price")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / f"{safe_name(product)}_day_{day}_normalized_orderbook.png", dpi=170)
        plt.close()


def plot_trade_distributions(trades_ctx: pd.DataFrame) -> None:
    for product, g in trades_ctx.groupby("product"):
        if g.empty:
            continue

        plt.figure(figsize=(10, 5))
        plt.hist(g["price"].dropna(), bins=40)
        plt.title(f"{product}: trade price distribution")
        plt.xlabel("trade price")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(OUT / f"{safe_name(product)}_trade_price_hist.png", dpi=160)
        plt.close()

        if g["trade_minus_mid"].notna().any():
            plt.figure(figsize=(10, 5))
            plt.hist(g["trade_minus_mid"].dropna(), bins=40)
            plt.axvline(0, linewidth=1)
            plt.title(f"{product}: trade price minus mid")
            plt.xlabel("trade price - mid_price")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(OUT / f"{safe_name(product)}_trade_minus_mid_hist.png", dpi=160)
            plt.close()


def make_markdown_report(prices: pd.DataFrame, trades_ctx: pd.DataFrame) -> None:
    lines = []
    lines.append("# Round 1 Data Analysis Report\n")
    lines.append("Generated by `analyse_round1.py`.\n")

    lines.append("## Products found\n")
    for p in sorted(prices["product"].dropna().unique()):
        lines.append(f"- {p}")
    lines.append("")

    lines.append("## Key interpretation guide\n")
    lines.append("- `spread = best_ask - best_bid`.")
    lines.append("- `bid_depth` and `ask_depth` are total visible volume across top 3 levels.")
    lines.append("- `depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth)`.")
    lines.append("- `trade_minus_mid > 0` means trade happened above observed mid.")
    lines.append("- For fixed-value products, compare trades/order levels to known fair value.")
    lines.append("- For moving products, normalize prices by `mid_price` or your own fair estimate.\n")

    overall = pd.read_csv(OUT / "overall_product_summary.csv")
    lines.append("## Overall product summary\n")
    lines.append(overall.to_markdown(index=False))
    lines.append("")

    trade_summary = pd.read_csv(OUT / "trade_summary_by_day_product.csv")
    lines.append("## Trade summary by day/product\n")
    lines.append(trade_summary.to_markdown(index=False))
    lines.append("")

    pred = pd.read_csv(OUT / "simple_predictability_checks.csv")
    lines.append("## Simple predictability checks\n")
    lines.append(pred.to_markdown(index=False))
    lines.append("")

    lines.append("## Files generated\n")
    for p in sorted(OUT.iterdir()):
        if p.name != "report.md":
            lines.append(f"- `{p.name}`")

    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s))


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    prices = load_prices()
    trades = load_trades()

    prices.to_csv(OUT / "clean_prices_all.csv", index=False)
    trades.to_csv(OUT / "clean_trades_all.csv", index=False)

    save_basic_summaries(prices, trades)
    trades_ctx = attach_trade_context(prices, trades)
    analyse_predictability(prices)

    plot_mid_and_spread(prices)
    plot_orderbook_cloud(prices, trades_ctx)
    plot_normalized_orderbook(prices, trades_ctx)
    plot_trade_distributions(trades_ctx)

    make_markdown_report(prices, trades_ctx)

    print(f"Done. Open: {OUT / 'report.md'}")
    print(f"Generated {len(list(OUT.iterdir()))} files in {OUT.resolve()}")


if __name__ == "__main__":
    main()
