#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK3 策略首秀：用均线交叉反应市场趋势变化

本脚本复用本仓库 TASK2 中已经保存的 A 股日行情 CSV，完成双均线交叉策略的
信号识别、逐日回测、绩效评价、图表、Notebook、README 和 PDF 报告生成。

安全说明：
- 不写入真实 Tushare token、MCP URL、.env 内容或本机绝对路径。
- 如需重新获取行情，可在本地另行设置环境变量 TUSHARE_TOKEN；本脚本默认复用
  已保存 CSV，保证作业结果可复现。
"""

from __future__ import annotations

import math
import json
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TASK2_DIR = PROJECT_DIR / "TASK2_jane"
DATA_DIR = BASE_DIR / "data"
FIG_DIR = BASE_DIR / "figures"
RESULT_DIR = BASE_DIR / "results"

PDF_PATH = BASE_DIR / "jane+TASK3.pdf"
README_PATH = BASE_DIR / "README.md"
NB_PATH = BASE_DIR / "task3_ma_cross_strategy.ipynb"
DASHBOARD_PATH = BASE_DIR / "ma_cross_dashboard.html"

INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0003
SLIPPAGE_RATE = 0.0002
RISK_FREE_RATE = 0.025
TRADING_DAYS = 252
MAIN_CODE = "300750.SZ"
MAIN_PARAMS = (5, 15)
PARAM_SETS = [(5, 15), (5, 20), (10, 30), (11, 35)]

STOCKS = [
    {"name": "贵州茅台", "code": "600519.SH", "source_csv": "600519_SH_daily_data.csv"},
    {"name": "宁德时代", "code": "300750.SZ", "source_csv": "300750_SZ_daily_data.csv"},
    {"name": "招商银行", "code": "600036.SH", "source_csv": "600036_SH_daily_data.csv"},
]

REQUIRED_COLUMNS = ["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]

FIG_PRICE = FIG_DIR / "figure1_price_ma_signals.png"
FIG_NAV = FIG_DIR / "figure2_strategy_vs_benchmark.png"
FIG_DRAWDOWN = FIG_DIR / "figure3_drawdown_curve.png"
FIG_COMPARE = FIG_DIR / "figure4_performance_comparison.png"


def choose_chinese_font() -> tuple[str, str | None]:
    candidates = [
        ("Songti SC", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("STSong", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
        ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
        ("Arial Unicode MS", "/Library/Fonts/Arial Unicode.ttf"),
        ("Noto Sans CJK SC", None),
        ("SimSun", None),
    ]
    available = {font.name for font in fm.fontManager.ttflist}
    for name, path in candidates:
        if path and Path(path).exists():
            return name, path
        if name in available:
            return name, None
    return "DejaVu Sans", None


CHINESE_FONT_NAME, CHINESE_FONT_PATH = choose_chinese_font()
plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def register_pdf_font() -> str:
    if CHINESE_FONT_PATH and Path(CHINESE_FONT_PATH).exists():
        try:
            pdfmetrics.registerFont(TTFont("ReportSong", CHINESE_FONT_PATH))
            return "ReportSong"
        except Exception:
            pass
    return "Helvetica"


def ensure_dirs() -> None:
    for folder in [DATA_DIR, FIG_DIR, RESULT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, REQUIRED_COLUMNS].copy()
    date_text = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df["trade_date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    for col in REQUIRED_COLUMNS:
        if col != "trade_date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("trade_date").dropna(subset=["trade_date"]).reset_index(drop=True)


def load_all_data() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    loaded: dict[str, pd.DataFrame] = {}
    for stock in STOCKS:
        source = TASK2_DIR / stock["source_csv"]
        target = DATA_DIR / stock["source_csv"]
        if not source.exists():
            raise FileNotFoundError(f"缺少本地行情 CSV：TASK2_jane/{stock['source_csv']}")
        shutil.copy2(source, target)
        df = normalize_daily_df(pd.read_csv(target))
        df["stock_name"] = stock["name"]
        df["ts_code"] = stock["code"]
        loaded[stock["code"]] = df
        df.to_csv(target, index=False, encoding="utf-8-sig")
    return loaded


def quality_checks(df: pd.DataFrame) -> dict[str, str]:
    price_cols = ["open", "high", "low", "close"]
    missing = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    duplicate_dates = int(df["trade_date"].duplicated().sum())
    sorted_ok = bool(df["trade_date"].is_monotonic_increasing)
    non_positive_prices = int((df[price_cols] <= 0).sum().sum())
    negative_volume_amount = int(((df[["vol", "amount"]] < 0).sum()).sum())
    ohlc_bad = int(
        (
            ~(
                (df["high"] >= df["open"])
                & (df["high"] >= df["close"])
                & (df["low"] <= df["open"])
                & (df["low"] <= df["close"])
                & (df["high"] >= df["low"])
            )
        ).sum()
    )
    return {
        "记录数": f"{len(df)}",
        "日期范围": f"{df['trade_date'].min().date()} 至 {df['trade_date'].max().date()}",
        "缺失值": str(missing),
        "重复日期": str(duplicate_dates),
        "日期升序": "是" if sorted_ok else "否",
        "非正价格": str(non_positive_prices),
        "负成交量/成交额": str(negative_volume_amount),
        "OHLC 异常": str(ohlc_bad),
    }


def add_signals(df: pd.DataFrame, short_window: int, long_window: int) -> pd.DataFrame:
    out = df.sort_values("trade_date").reset_index(drop=True).copy()
    out["ma_short"] = out["close"].rolling(short_window, min_periods=short_window).mean()
    out["ma_long"] = out["close"].rolling(long_window, min_periods=long_window).mean()
    prev_short = out["ma_short"].shift(1)
    prev_long = out["ma_long"].shift(1)
    out["golden_cross"] = (out["ma_short"] > out["ma_long"]) & (prev_short <= prev_long)
    out["death_cross"] = (out["ma_short"] < out["ma_long"]) & (prev_short >= prev_long)
    out["signal"] = np.select([out["golden_cross"], out["death_cross"]], [1, -1], default=0)
    return out


def run_backtest(
    df: pd.DataFrame,
    short_window: int,
    long_window: int,
    initial_capital: float = INITIAL_CAPITAL,
    commission_rate: float = COMMISSION_RATE,
    slippage_rate: float = SLIPPAGE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str]]:
    work = add_signals(df, short_window, long_window)
    cash = initial_capital
    shares = 0
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []
    entry_cash_flow = 0.0

    for _, row in work.iterrows():
        date = row["trade_date"]
        close = float(row["close"])
        direction = ""
        trade_price = np.nan
        trade_shares = 0
        fee = 0.0
        slippage = 0.0
        cash_before = cash
        shares_before = shares

        if shares == 0 and bool(row["golden_cross"]):
            effective_price = close * (1 + slippage_rate)
            affordable = math.floor(cash / (effective_price * (1 + commission_rate)) / 100) * 100
            if affordable > 0:
                trade_value = affordable * effective_price
                fee = trade_value * commission_rate
                slippage = affordable * close * slippage_rate
                cash -= trade_value + fee
                shares += affordable
                direction = "买入"
                trade_price = effective_price
                trade_shares = affordable
                entry_cash_flow = trade_value + fee
        elif shares > 0 and bool(row["death_cross"]):
            effective_price = close * (1 - slippage_rate)
            trade_value = shares * effective_price
            fee = trade_value * commission_rate
            slippage = shares * close * slippage_rate
            cash += trade_value - fee
            direction = "卖出"
            trade_price = effective_price
            trade_shares = shares
            if entry_cash_flow > 0:
                trade_pnl = trade_value - fee - entry_cash_flow
            else:
                trade_pnl = np.nan
            shares = 0
            entry_cash_flow = 0.0
        else:
            trade_pnl = np.nan

        total_asset = cash + shares * close
        position_value = shares * close
        position_ratio = position_value / total_asset if total_asset else 0.0

        if direction:
            if direction == "买入":
                trade_pnl = np.nan
            trade_rows.append(
                {
                    "trade_date": date,
                    "stock_name": row["stock_name"],
                    "ts_code": row["ts_code"],
                    "direction": direction,
                    "signal": "金叉" if direction == "买入" else "死叉",
                    "close": close,
                    "trade_price": trade_price,
                    "shares": trade_shares,
                    "fee": fee,
                    "slippage_cost": slippage,
                    "cash_before": cash_before,
                    "shares_before": shares_before,
                    "cash_after": cash,
                    "shares_after": shares,
                    "total_asset": total_asset,
                    "trade_pnl": trade_pnl,
                }
            )

        equity_rows.append(
            {
                "trade_date": date,
                "stock_name": row["stock_name"],
                "ts_code": row["ts_code"],
                "close": close,
                "ma_short": row["ma_short"],
                "ma_long": row["ma_long"],
                "golden_cross": bool(row["golden_cross"]),
                "death_cross": bool(row["death_cross"]),
                "signal": int(row["signal"]),
                "cash": cash,
                "shares": shares,
                "position_ratio": position_ratio,
                "total_asset": total_asset,
            }
        )

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    equity["strategy_nav"] = equity["total_asset"] / initial_capital
    equity["strategy_return"] = equity["strategy_nav"].pct_change().fillna(0.0)
    equity["benchmark_nav"] = equity["close"] / equity["close"].iloc[0]
    equity["benchmark_return"] = equity["benchmark_nav"].pct_change().fillna(0.0)
    equity["cummax_nav"] = equity["strategy_nav"].cummax()
    equity["drawdown"] = equity["strategy_nav"] / equity["cummax_nav"] - 1

    metrics = calculate_metrics(equity, trades, short_window, long_window)
    return equity, trades, metrics


def calculate_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    short_window: int,
    long_window: int,
) -> dict[str, float | int | str]:
    days = max(len(equity), 1)
    total_return = float(equity["strategy_nav"].iloc[-1] - 1)
    benchmark_return = float(equity["benchmark_nav"].iloc[-1] - 1)
    annual_return = float((1 + total_return) ** (TRADING_DAYS / days) - 1) if total_return > -1 else -1.0
    daily_rf = (1 + RISK_FREE_RATE) ** (1 / TRADING_DAYS) - 1
    excess_daily = equity["strategy_return"] - daily_rf
    if equity["strategy_return"].std(ddof=1) > 0:
        sharpe = float(excess_daily.mean() / equity["strategy_return"].std(ddof=1) * np.sqrt(TRADING_DAYS))
    else:
        sharpe = 0.0
    max_drawdown = float(equity["drawdown"].min())
    sells = trades[trades["direction"] == "卖出"].copy() if not trades.empty else pd.DataFrame()
    completed = sells["trade_pnl"].dropna() if not sells.empty and "trade_pnl" in sells else pd.Series(dtype=float)
    wins = completed[completed > 0]
    losses = completed[completed < 0]
    win_rate = float(len(wins) / len(completed)) if len(completed) else 0.0
    if len(losses) and len(wins):
        profit_loss_ratio = float(wins.mean() / abs(losses.mean()))
    elif len(wins):
        profit_loss_ratio = float("inf")
    else:
        profit_loss_ratio = 0.0
    return {
        "stock_name": str(equity["stock_name"].iloc[0]),
        "ts_code": str(equity["ts_code"].iloc[0]),
        "short_window": int(short_window),
        "long_window": int(long_window),
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_loss_ratio": profit_loss_ratio,
        "trade_count": int(len(trades)),
        "completed_trades": int(len(completed)),
        "signal_density": float((equity["signal"] != 0).mean()),
        "benchmark_return": benchmark_return,
        "excess_return": total_return - benchmark_return,
        "final_asset": float(equity["total_asset"].iloc[-1]),
    }


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def ratio(value: float) -> str:
    if math.isinf(value):
        return "无亏损"
    return f"{value:.2f}"


def make_all_backtests(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    metrics_rows: list[dict[str, float | int | str]] = []
    equity_map: dict[str, pd.DataFrame] = {}
    trade_map: dict[str, pd.DataFrame] = {}
    for stock in STOCKS:
        df = data[stock["code"]]
        for short_window, long_window in PARAM_SETS:
            equity, trades, metrics = run_backtest(df, short_window, long_window)
            key = f"{stock['code']}_{short_window}_{long_window}"
            equity_map[key] = equity
            trade_map[key] = trades
            metrics_rows.append(metrics)
            equity.to_csv(RESULT_DIR / f"equity_{stock['code'].replace('.', '_')}_{short_window}_{long_window}.csv", index=False, encoding="utf-8-sig")
            trades.to_csv(RESULT_DIR / f"trades_{stock['code'].replace('.', '_')}_{short_window}_{long_window}.csv", index=False, encoding="utf-8-sig")
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(RESULT_DIR / "task3_performance_summary.csv", index=False, encoding="utf-8-sig")
    return metrics_df, equity_map, trade_map


def plot_price_signals(equity: pd.DataFrame, trades: pd.DataFrame, short_window: int, long_window: int) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    ax.plot(equity["trade_date"], equity["close"], label="收盘价", color="#3949ab", linewidth=1.5)
    ax.plot(equity["trade_date"], equity["ma_short"], label=f"MA{short_window}", color="#f28e2b", linewidth=1.4)
    ax.plot(equity["trade_date"], equity["ma_long"], label=f"MA{long_window}", color="#59a14f", linewidth=1.4)
    if not trades.empty:
        buys = trades[trades["direction"] == "买入"]
        sells = trades[trades["direction"] == "卖出"]
        ax.scatter(buys["trade_date"], buys["close"], marker="^", s=72, color="#d62728", label="买入（金叉）", zorder=4)
        ax.scatter(sells["trade_date"], sells["close"], marker="v", s=72, color="#1f77b4", label="卖出（死叉）", zorder=4)
    stock_name = str(equity["stock_name"].iloc[0])
    ax.set_title(f"{stock_name} 5/15 双均线交叉信号", fontsize=15, pad=14)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("价格（元）")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(ncol=3, frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_PRICE, bbox_inches="tight")
    plt.close(fig)


def plot_nav(equity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=180)
    ax.plot(equity["trade_date"], equity["strategy_nav"], label="双均线策略净值", color="#d95f02", linewidth=2)
    ax.plot(equity["trade_date"], equity["benchmark_nav"], label="买入持有基准", color="#1b9e77", linewidth=2)
    ax.axhline(1, color="#555555", linewidth=0.8, alpha=0.6)
    ax.set_title("策略净值与买入持有基准对比", fontsize=15, pad=14)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("净值")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_NAV, bbox_inches="tight")
    plt.close(fig)


def plot_drawdown(equity: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.2), dpi=180)
    ax.fill_between(equity["trade_date"], equity["drawdown"] * 100, 0, color="#4daf4a", alpha=0.28)
    ax.plot(equity["trade_date"], equity["drawdown"] * 100, color="#238b45", linewidth=1.6)
    ax.set_title("策略回撤曲线", fontsize=15, pad=14)
    ax.set_xlabel("交易日期")
    ax.set_ylabel("回撤（%）")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG_DRAWDOWN, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(metrics_df: pd.DataFrame) -> None:
    subset = metrics_df.copy()
    subset["label"] = subset["stock_name"] + "\n" + subset["short_window"].astype(str) + "/" + subset["long_window"].astype(str)
    x = np.arange(len(subset))
    width = 0.28
    fig, axes = plt.subplots(2, 1, figsize=(12, 8.6), dpi=180, sharex=True)
    axes[0].bar(x - width, subset["total_return"] * 100, width, label="累计回报", color="#e15759")
    axes[0].bar(x, subset["excess_return"] * 100, width, label="超额收益", color="#76b7b2")
    axes[0].bar(x + width, subset["sharpe"], width, label="夏普比率", color="#f28e2b")
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    axes[0].set_ylabel("收益率（%）/ 夏普")
    axes[0].set_title("多股票与多参数绩效对比", fontsize=15, pad=12)
    axes[0].legend(ncol=3, frameon=False)
    axes[0].grid(True, axis="y", linestyle="--", alpha=0.28)

    axes[1].bar(x, subset["max_drawdown"] * 100, color="#59a14f")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("最大回撤（%）")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(subset["label"], rotation=35, ha="right")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.28)
    fig.tight_layout()
    fig.savefig(FIG_COMPARE, bbox_inches="tight")
    plt.close(fig)


def make_figures(equity_map: dict[str, pd.DataFrame], trade_map: dict[str, pd.DataFrame], metrics_df: pd.DataFrame) -> None:
    key = f"{MAIN_CODE}_{MAIN_PARAMS[0]}_{MAIN_PARAMS[1]}"
    main_equity = equity_map[key]
    main_trades = trade_map[key]
    plot_price_signals(main_equity, main_trades, *MAIN_PARAMS)
    plot_nav(main_equity)
    plot_drawdown(main_equity)
    plot_comparison(metrics_df)


def table_data_from_df(df: pd.DataFrame, columns: list[str], header_map: dict[str, str]) -> list[list[str]]:
    rows = [[header_map.get(col, col) for col in columns]]
    for _, row in df.iterrows():
        current = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                if col in {"total_return", "annual_return", "max_drawdown", "win_rate", "benchmark_return", "excess_return"}:
                    current.append(pct(value))
                elif col == "profit_loss_ratio":
                    current.append(ratio(value))
                elif col in {"sharpe"}:
                    current.append(f"{value:.2f}")
                else:
                    current.append(f"{value:.2f}")
            else:
                current.append(str(value))
        rows.append(current)
    return rows


def add_table(story: list, rows: list[list[str]], col_widths: list[float], font_name: str, font_size: int = 8) -> None:
    table = Table(rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f3f7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fbfdff")]),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def fig_block(path: Path, caption: str, note: str, styles: dict[str, ParagraphStyle]) -> KeepTogether:
    return KeepTogether(
        [
            Image(str(path), width=15.4 * cm, height=8.1 * cm),
            p(caption, styles["caption"]),
            p(note, styles["body"]),
            Spacer(1, 0.18 * cm),
        ]
    )


def build_pdf(metrics_df: pd.DataFrame, equity_map: dict[str, pd.DataFrame], trade_map: dict[str, pd.DataFrame], data: dict[str, pd.DataFrame]) -> None:
    font_name = register_pdf_font()
    base_styles = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}
    styles["title"] = ParagraphStyle(
        "ChineseTitle",
        parent=base_styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=0,
    )
    styles["heading"] = ParagraphStyle(
        "ChineseHeading",
        parent=base_styles["Heading2"],
        fontName=font_name,
        fontSize=13,
        leading=18,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
        textColor=colors.HexColor("#1f2937"),
    )
    styles["body"] = ParagraphStyle(
        "ChineseBody",
        parent=base_styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=15.75,
        firstLineIndent=21,
        alignment=TA_JUSTIFY,
        spaceBefore=0,
        spaceAfter=0,
    )
    styles["caption"] = ParagraphStyle(
        "ChineseCaption",
        parent=base_styles["BodyText"],
        fontName=font_name,
        fontSize=10.5,
        leading=15.75,
        alignment=TA_CENTER,
        spaceBefore=0,
        spaceAfter=0,
        textColor=colors.HexColor("#374151"),
    )
    styles["code"] = ParagraphStyle(
        "ChineseCode",
        parent=base_styles["Code"],
        fontName=font_name,
        fontSize=8.5,
        leading=11,
        leftIndent=12,
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=0,
    )

    key = f"{MAIN_CODE}_{MAIN_PARAMS[0]}_{MAIN_PARAMS[1]}"
    main_equity = equity_map[key]
    main_trades = trade_map[key]
    main_metrics = metrics_df[
        (metrics_df["ts_code"] == MAIN_CODE)
        & (metrics_df["short_window"] == MAIN_PARAMS[0])
        & (metrics_df["long_window"] == MAIN_PARAMS[1])
    ].iloc[0]
    best_return = metrics_df.sort_values("total_return", ascending=False).iloc[0]
    lowest_mdd = metrics_df.sort_values("max_drawdown", ascending=False).iloc[0]

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.7 * cm,
        title="jane+TASK3",
        author="jane",
    )
    story: list = []

    story.append(p("TASK3 策略首秀：用均线交叉反应市场趋势变化", styles["title"]))
    story.append(Spacer(1, 0.35 * cm))
    story.append(p("一、任务背景与策略思想", styles["heading"]))
    story.append(
        p(
            "本次作业围绕双均线交叉策略展开。短期均线反映较近一段时间的价格变化，反应更快；长期均线反映更长窗口的平均价格，曲线更平滑。"
            "当短期均线和长期均线发生交叉时，说明市场短期力量与中长期趋势的相对关系发生变化，因此可以把交叉点作为趋势可能切换的观察信号。",
            styles["body"],
        )
    )
    story.append(
        p(
            "金叉指短期均线从下方向上穿越长期均线，通常作为买入信号；死叉指短期均线从上方向下穿越长期均线，通常作为卖出信号。"
            "需要特别说明的是，交易信号不是“短期均线大于长期均线”的持续状态，而是穿越发生的那一个交易日。"
            "因此本报告用 MA_short_t > MA_long_t 且 MA_short_(t-1) <= MA_long_(t-1) 判断金叉，用相反条件判断死叉，避免把多头排列期间每天都误认为买入。",
            styles["body"],
        )
    )

    story.append(Spacer(1, 0.18 * cm))
    story.append(p("二、数据来源与参数设置", styles["heading"]))
    story.append(
        p(
            "数据复用本仓库 TASK2 中已经保存的日行情 CSV，包含贵州茅台、宁德时代和招商银行三只 A 股。"
            "字段包括交易日期、开盘价、最高价、最低价、收盘价、前收盘价、涨跌额、涨跌幅、成交量和成交额。"
            "这样可以保证结果可复现，也避免在作业文件中写入任何真实 Tushare token 或本地凭据。",
            styles["body"],
        )
    )
    quality_rows = [["股票", "代码", "记录数", "日期范围", "缺失值", "重复日期", "OHLC 异常"]]
    for stock in STOCKS:
        qc = quality_checks(data[stock["code"]])
        quality_rows.append([stock["name"], stock["code"], qc["记录数"], qc["日期范围"], qc["缺失值"], qc["重复日期"], qc["OHLC 异常"]])
    add_table(story, quality_rows, [2.0 * cm, 2.4 * cm, 1.6 * cm, 4.2 * cm, 1.6 * cm, 1.8 * cm, 1.8 * cm], font_name, 8)
    story.append(
        p(
            "主实验采用课堂截图要求的 5 日短均线和 15 日长均线。扩展实验同时比较 5/20、10/30、11/35 三组参数，并在三只股票上重复回测。",
            styles["body"],
        )
    )

    story.append(Spacer(1, 0.18 * cm))
    story.append(p("三、Python 实现与回测流程", styles["heading"]))
    story.append(
        p(
            "程序先按交易日期升序排列数据，检查缺失值、重复日期和 OHLC 价格逻辑，再计算短均线与长均线。"
            "回测按交易日逐日推进：当前空仓且当天发生金叉时买入，当前持仓且当天发生死叉时卖出，没有新信号则保持原仓位。"
            "每个交易日记录现金、股数、仓位比例和总资产，保证不使用未来数据。",
            styles["body"],
        )
    )
    story.append(
        p(
            "交易假设为初始资金 100000 元，金叉时按 A 股 100 股整数手尽量满仓买入，死叉时全部卖出；手续费设为 0.03%，滑点设为 0.02%。"
            "为贴合课程入门作业，交易价格采用当天收盘价执行的简化假设，实际交易中应改为下一交易日开盘价或更细颗粒度成交价。",
            styles["body"],
        )
    )
    story.append(
        p(
            "核心代码逻辑可概括为：先用 rolling(window) 得到 MA_short 和 MA_long，再用当日与前一日的均线相对位置识别金叉/死叉；"
            "随后在循环中根据当前是否持仓决定买入、卖出或保持，最后由 total_asset / initial_capital 得到策略净值。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("四、单只股票回测结果分析", styles["heading"]))
    story.append(
        p(
            f"主实验对象为 {main_metrics['stock_name']}（{MAIN_CODE}），参数为 MA{MAIN_PARAMS[0]}/MA{MAIN_PARAMS[1]}。"
            f"回测期内策略累计回报为 {pct(float(main_metrics['total_return']))}，年化收益率为 {pct(float(main_metrics['annual_return']))}，"
            f"最大回撤为 {pct(float(main_metrics['max_drawdown']))}，夏普比率为 {float(main_metrics['sharpe']):.2f}。"
            f"买入持有基准收益为 {pct(float(main_metrics['benchmark_return']))}，策略超额收益为 {pct(float(main_metrics['excess_return']))}。",
            styles["body"],
        )
    )
    story.append(
        fig_block(
            FIG_PRICE,
            f"图1 {main_metrics['stock_name']} 5/15 双均线交叉信号",
            "图1展示了收盘价、5日均线、15日均线以及买卖点。买点只出现在短均线上穿长均线的金叉日，卖点只出现在短均线下穿长均线的死叉日；在两条均线持续保持同一相对位置时，策略只是维持原仓位。",
            styles,
        )
    )
    story.append(
        fig_block(
            FIG_NAV,
            "图2 策略净值与买入持有基准对比",
            "图2用于观察策略是否优于简单持有。若策略净值在下跌阶段减少持仓，通常能降低亏损；但在震荡或快速反转行情中，均线滞后也可能导致错过反弹或反复交易。",
            styles,
        )
    )
    story.append(
        fig_block(
            FIG_DRAWDOWN,
            "图3 策略回撤曲线",
            "图3反映从历史净值高点回落的幅度。最大回撤越深，说明投资者在最不利阶段需要承受的浮亏越大，因此它比单纯收益率更能体现风险。",
            styles,
        )
    )

    story.append(PageBreak())
    story.append(p("五、多股票与多参数对比实验", styles["heading"]))
    display_cols = [
        "stock_name",
        "ts_code",
        "short_window",
        "long_window",
        "total_return",
        "annual_return",
        "sharpe",
        "max_drawdown",
        "win_rate",
        "profit_loss_ratio",
        "trade_count",
        "excess_return",
    ]
    header_map = {
        "stock_name": "股票",
        "ts_code": "代码",
        "short_window": "短均线",
        "long_window": "长均线",
        "total_return": "累计回报",
        "annual_return": "年化收益",
        "sharpe": "夏普",
        "max_drawdown": "最大回撤",
        "win_rate": "胜率",
        "profit_loss_ratio": "盈亏比",
        "trade_count": "交易次数",
        "excess_return": "超额收益",
    }
    display_metrics = metrics_df.sort_values(["stock_name", "short_window", "long_window"]).reset_index(drop=True)
    add_table(
        story,
        table_data_from_df(display_metrics, display_cols, header_map),
        [1.55 * cm, 1.8 * cm, 1.15 * cm, 1.15 * cm, 1.45 * cm, 1.45 * cm, 1.05 * cm, 1.45 * cm, 1.25 * cm, 1.25 * cm, 1.25 * cm, 1.45 * cm],
        font_name,
        6,
    )
    story.append(
        p(
            f"从汇总表看，累计回报最高的是 {best_return['stock_name']}（{best_return['ts_code']}）"
            f"的 {int(best_return['short_window'])}/{int(best_return['long_window'])} 组合，累计回报为 {pct(float(best_return['total_return']))}。"
            f"最大回撤最小的是 {lowest_mdd['stock_name']}（{lowest_mdd['ts_code']}）"
            f"的 {int(lowest_mdd['short_window'])}/{int(lowest_mdd['long_window'])} 组合，最大回撤为 {pct(float(lowest_mdd['max_drawdown']))}。"
            "这说明双均线策略对标的和参数都比较敏感，不能只凭一组参数判断策略好坏。",
            styles["body"],
        )
    )
    story.append(
        fig_block(
            FIG_COMPARE,
            "图4 多股票与多参数绩效对比",
            "图4把累计回报、超额收益、夏普比率和最大回撤放在一起比较。趋势较清晰且回撤受控的组合更适合双均线策略；若收益不高但交易次数较多，说明震荡行情中的假信号和交易成本可能侵蚀收益。",
            styles,
        )
    )

    story.append(p("六、策略评价指标解释", styles["heading"]))
    story.append(
        p(
            "累计回报衡量整个回测期间策略最终赚了多少，公式可以写为期末净值减 1。年化收益率把不同长度的回测结果标准化到每年，便于横向比较。"
            "最大回撤 MDD 衡量从历史峰值到之后谷底的最大跌幅，可写为 max((峰值 - 谷底) / 峰值)，是衡量最坏阶段风险的重要指标。"
            "夏普比率 Sharpe = (Rp - Rf) / σp，本报告默认无风险利率 2.5%、一年 252 个交易日，用它观察每承担一单位波动获得多少超额收益。",
            styles["body"],
        )
    )
    story.append(
        p(
            "胜率表示盈利交易占完成交易的比例，盈亏比表示平均盈利与平均亏损的比值。胜率高不一定代表策略好，因为少数大亏也可能抵消多次小赚；"
            "盈亏比高也不一定充分，因为交易次数太少时稳定性不足。因此本报告把收益、回撤、夏普、胜率、交易次数和超额收益一起观察。",
            styles["body"],
        )
    )

    story.append(Spacer(1, 0.18 * cm))
    story.append(p("七、总结与心得", styles["heading"]))
    story.append(
        p(
            "双均线策略的优点是规则简单、容易解释、容易实现，在趋势较明显的行情中能帮助跟随主要方向，并在死叉后减少继续持有的风险。"
            "它的缺点也很明显：均线本质上是滞后指标，在震荡行情中容易产生频繁假信号；不同均线周期会带来不同结果，参数敏感；手续费和滑点会持续侵蚀收益。",
            styles["body"],
        )
    )
    story.append(
        p(
            "我的应用心得是，均线交叉适合作为趋势过滤器，而不适合单独作为完整交易系统。实际使用时应结合止损、仓位管理、交易成本、市场环境判断和更多风控规则。"
            "尤其在课程截图强调的策略四要素中，入场只是第一步，出场、仓位和风险控制同样决定了策略能否长期稳定。",
            styles["body"],
        )
    )

    story.append(PageBreak())
    story.append(p("附录：核心 Python 代码片段", styles["heading"]))
    code_text = """
df["ma_short"] = df["close"].rolling(short_window).mean()
df["ma_long"] = df["close"].rolling(long_window).mean()
golden = (df["ma_short"] > df["ma_long"]) & (df["ma_short"].shift(1) <= df["ma_long"].shift(1))
death = (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))

for each trading day:
    if no_position and golden_cross:
        buy with available cash
    elif holding_position and death_cross:
        sell all shares
    else:
        keep current position
    total_asset = cash + shares * close
"""
    for line in code_text.strip().splitlines():
        story.append(p(line.replace(" ", "&nbsp;"), styles["code"]))

    def footer(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawCentredString(A4[0] / 2, 0.9 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def build_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# TASK3 策略首秀：用均线交叉反应市场趋势变化\n\n本 Notebook 由 `task3_ma_cross_strategy.py` 自动生成，主要结果请运行脚本复现。"),
        nbf.v4.new_code_cell(
            "from task3_ma_cross_strategy import load_all_data, make_all_backtests, make_figures\n"
            "data = load_all_data()\n"
            "metrics_df, equity_map, trade_map = make_all_backtests(data)\n"
            "make_figures(equity_map, trade_map, metrics_df)\n"
            "metrics_df"
        ),
        nbf.v4.new_markdown_cell(
            "核心信号定义：金叉是短期均线从下方向上穿越长期均线，死叉是短期均线从上方向下穿越长期均线。信号是穿越发生的时刻，不是短均线持续大于或小于长均线的状态。"
        ),
    ]
    with NB_PATH.open("w", encoding="utf-8") as f:
        nbf.write(nb, f)


def build_dashboard(data: dict[str, pd.DataFrame], metrics_df: pd.DataFrame) -> None:
    stock_payload = []
    for stock in STOCKS:
        df = data[stock["code"]].copy()
        stock_payload.append(
            {
                "name": stock["name"],
                "code": stock["code"],
                "rows": [
                    {
                        "date": row.trade_date.strftime("%Y-%m-%d"),
                        "open": float(row.open),
                        "high": float(row.high),
                        "low": float(row.low),
                        "close": float(row.close),
                        "vol": float(row.vol),
                    }
                    for row in df.itertuples(index=False)
                ],
            }
        )
    compare_payload = []
    for row in metrics_df.itertuples(index=False):
        compare_payload.append(
            {
                "label": f"{row.stock_name}\\n{row.short_window}/{row.long_window}",
                "stock": row.stock_name,
                "code": row.ts_code,
                "short": int(row.short_window),
                "long": int(row.long_window),
                "totalReturn": float(row.total_return),
                "annualReturn": float(row.annual_return),
                "sharpe": float(row.sharpe),
                "maxDrawdown": float(row.max_drawdown),
                "winRate": float(row.win_rate),
                "tradeCount": int(row.trade_count),
                "excessReturn": float(row.excess_return),
            }
        )
    payload = {
        "stocks": stock_payload,
        "comparison": compare_payload,
        "defaults": {
            "stock": MAIN_CODE,
            "short": MAIN_PARAMS[0],
            "long": MAIN_PARAMS[1],
            "initialCapital": INITIAL_CAPITAL,
            "commissionRate": COMMISSION_RATE,
            "slippageRate": SLIPPAGE_RATE,
            "riskFreeRate": RISK_FREE_RATE,
            "tradingDays": TRADING_DAYS,
        },
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>双均线策略回测看板</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
      color: #222;
      background: #f6f7fb;
    }}
    header {{
      height: 58px;
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 0 22px;
      background: #fff;
      border-bottom: 1px solid #e7eaf0;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    header h1 {{ font-size: 20px; margin: 0; font-weight: 700; }}
    header .pill {{ padding: 6px 12px; background: #eef0f4; border-radius: 999px; font-size: 13px; color: #555; }}
    main {{ display: grid; grid-template-columns: 300px 1fr; min-height: calc(100vh - 58px); }}
    aside {{
      background: #fff;
      border-right: 1px solid #e7eaf0;
      padding: 20px 18px;
      overflow-y: auto;
    }}
    section.content {{ padding: 20px; }}
    .group {{ margin-bottom: 22px; }}
    .group h2 {{ font-size: 15px; margin: 0 0 12px; color: #4a5568; }}
    label {{ display: block; font-size: 13px; color: #5d6675; margin: 10px 0 6px; }}
    select, input[type="date"], input[type="number"] {{
      width: 100%;
      border: 1px solid #d9dee8;
      border-radius: 8px;
      padding: 10px 11px;
      font-size: 14px;
      background: #fff;
    }}
    input[type="range"] {{ width: 100%; accent-color: #2f8ec4; }}
    .range-row {{ display: grid; grid-template-columns: 1fr 34px; gap: 10px; align-items: center; }}
    .range-value {{ color: #2f8ec4; font-weight: 700; text-align: right; }}
    .switch-row {{ display: flex; align-items: center; justify-content: space-between; margin: 12px 0; font-size: 14px; }}
    .switch-row input {{ width: 42px; height: 22px; accent-color: #2f8ec4; }}
    button {{
      width: 100%;
      border: 1px solid #d9dee8;
      background: #fff;
      border-radius: 8px;
      padding: 10px;
      font-weight: 700;
      cursor: pointer;
    }}
    .side-metrics {{ border-top: 1px solid #edf0f5; padding-top: 14px; }}
    .side-metrics p {{ margin: 8px 0; font-size: 14px; color: #4a5568; }}
    .side-metrics strong {{ color: #d8343f; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; margin-bottom: 18px; }}
    .card, .panel {{
      background: #fff;
      border: 1px solid #e8ebf2;
      border-radius: 8px;
      box-shadow: 0 1px 2px rgba(20, 30, 50, 0.03);
    }}
    .card {{ min-height: 96px; display: flex; flex-direction: column; justify-content: center; align-items: center; padding: 12px; }}
    .card .label {{ font-size: 13px; color: #6b7280; }}
    .card .value {{ font-size: 26px; font-weight: 800; margin-top: 8px; }}
    .card .note {{ font-size: 12px; color: #7b8494; margin-top: 6px; text-align: center; }}
    .red {{ color: #d8343f; }}
    .green {{ color: #0c9f6e; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .wide {{ grid-column: 1 / -1; }}
    .panel {{ padding: 16px 16px 12px; min-height: 320px; }}
    .panel h3 {{ margin: 0 0 12px; font-size: 16px; }}
    svg {{ width: 100%; height: 280px; overflow: visible; }}
    .wide svg {{ height: 330px; }}
    .legend {{ display: flex; gap: 16px; justify-content: center; color: #5b6370; font-size: 13px; margin-top: -4px; }}
    .legend span::before {{ content: ""; display: inline-block; width: 11px; height: 11px; border-radius: 50%; margin-right: 5px; vertical-align: -1px; background: var(--c); }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: none; border-bottom: 1px solid #e7eaf0; }}
      .cards {{ grid-template-columns: repeat(2, 1fr); }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>🎯 双均线策略回测看板</h1>
    <span class="pill" id="paramPill">SMA(5,15)</span>
  </header>
  <main>
    <aside>
      <div class="group">
        <h2>📌 标的选择</h2>
        <select id="stockSelect"></select>
        <label id="stockMeta"></label>
      </div>
      <div class="group">
        <h2>📅 时间窗口</h2>
        <label>起始日期</label>
        <input type="date" id="startDate">
        <label>结束日期</label>
        <input type="date" id="endDate">
      </div>
      <div class="group">
        <h2>📏 均线参数</h2>
        <label>短周期 SMA</label>
        <div class="range-row"><input type="range" id="shortRange" min="2" max="30" value="5"><span class="range-value" id="shortValue">5</span></div>
        <label>长周期 SMA</label>
        <div class="range-row"><input type="range" id="longRange" min="5" max="60" value="15"><span class="range-value" id="longValue">15</span></div>
      </div>
      <div class="group">
        <h2>💸 交易成本</h2>
        <div class="switch-row"><span>手续费</span><input type="checkbox" id="commissionToggle" checked></div>
        <div class="switch-row"><span>滑点</span><input type="checkbox" id="slippageToggle" checked></div>
        <label>初始资金</label>
        <input type="number" id="capitalInput" value="100000" min="10000" step="10000">
      </div>
      <button id="resetBtn">🔁 重置默认参数</button>
      <div class="side-metrics group">
        <h2>📊 策略指标</h2>
        <p>年化收益：<strong id="sideAnnual">-</strong></p>
        <p>夏普比率：<strong id="sideSharpe">-</strong></p>
        <p>最大回撤：<strong id="sideMdd">-</strong></p>
        <p>交易次数：<strong id="sideTrades">-</strong></p>
        <p>胜率：<strong id="sideWin">-</strong></p>
        <p>盈亏比：<strong id="sidePl">-</strong></p>
        <p>超额收益：<strong id="sideExcess">-</strong></p>
      </div>
    </aside>
    <section class="content">
      <div class="cards">
        <div class="card"><div class="label">年化收益率</div><div class="value" id="annualCard">-</div><div class="note" id="returnNote">-</div></div>
        <div class="card"><div class="label">夏普比率</div><div class="value" id="sharpeCard">-</div><div class="note" id="sharpeNote">-</div></div>
        <div class="card"><div class="label">最大回撤</div><div class="value green" id="mddCard">-</div><div class="note" id="mddNote">-</div></div>
        <div class="card"><div class="label">胜率</div><div class="value" id="winCard">-</div><div class="note" id="tradeNote">-</div></div>
      </div>
      <div class="grid">
        <div class="panel wide">
          <h3>策略净值 vs 买入持有基准</h3>
          <svg id="navChart"></svg>
          <div class="legend"><span style="--c:#d8343f">策略净值</span><span style="--c:#9aa1ad">买入持有</span></div>
        </div>
        <div class="panel">
          <h3>回撤（%）</h3>
          <svg id="drawdownChart"></svg>
        </div>
        <div class="panel">
          <h3>价格 + 均线 + 买卖点</h3>
          <svg id="priceChart"></svg>
          <div class="legend"><span style="--c:#555">收盘价</span><span style="--c:#f2a65a">短均线</span><span style="--c:#2f8ec4">长均线</span><span style="--c:#d8343f">买入</span><span style="--c:#0c9f6e">卖出</span></div>
        </div>
        <div class="panel wide">
          <h3>多股票与多参数绩效对比</h3>
          <svg id="compareChart"></svg>
        </div>
      </div>
    </section>
  </main>
  <script>
    const DATA = {payload_json};
    const els = Object.fromEntries(["stockSelect","stockMeta","startDate","endDate","shortRange","longRange","shortValue","longValue","commissionToggle","slippageToggle","capitalInput","resetBtn","paramPill","annualCard","returnNote","sharpeCard","sharpeNote","mddCard","mddNote","winCard","tradeNote","sideAnnual","sideSharpe","sideMdd","sideTrades","sideWin","sidePl","sideExcess","navChart","drawdownChart","priceChart","compareChart"].map(id => [id, document.getElementById(id)]));

    function pct(v) {{ return (v * 100).toFixed(2) + "%"; }}
    function num(v, n=3) {{ return Number.isFinite(v) ? v.toFixed(n) : "无亏损"; }}
    function dateStr(d) {{ return d.date; }}
    function cls(v) {{ return v >= 0 ? "red" : "green"; }}
    function stock() {{ return DATA.stocks.find(s => s.code === els.stockSelect.value); }}

    function ma(rows, window) {{
      const out = Array(rows.length).fill(null);
      let sum = 0;
      for (let i = 0; i < rows.length; i++) {{
        sum += rows[i].close;
        if (i >= window) sum -= rows[i - window].close;
        if (i >= window - 1) out[i] = sum / window;
      }}
      return out;
    }}

    function runBacktest(rows, shortW, longW, capital, useFee, useSlip) {{
      const shortMa = ma(rows, shortW);
      const longMa = ma(rows, longW);
      const feeRate = useFee ? DATA.defaults.commissionRate : 0;
      const slipRate = useSlip ? DATA.defaults.slippageRate : 0;
      let cash = capital, shares = 0, entry = 0;
      const equity = [], trades = [], pnls = [];
      for (let i = 0; i < rows.length; i++) {{
        const close = rows[i].close;
        const golden = i > 0 && shortMa[i] != null && longMa[i] != null && shortMa[i] > longMa[i] && shortMa[i - 1] <= longMa[i - 1];
        const death = i > 0 && shortMa[i] != null && longMa[i] != null && shortMa[i] < longMa[i] && shortMa[i - 1] >= longMa[i - 1];
        if (shares === 0 && golden) {{
          const price = close * (1 + slipRate);
          const buyShares = Math.floor(cash / (price * (1 + feeRate)) / 100) * 100;
          if (buyShares > 0) {{
            const value = buyShares * price, fee = value * feeRate;
            cash -= value + fee;
            shares = buyShares;
            entry = value + fee;
            trades.push({{date: rows[i].date, type: "buy", close, price, shares: buyShares}});
          }}
        }} else if (shares > 0 && death) {{
          const price = close * (1 - slipRate);
          const value = shares * price, fee = value * feeRate;
          const pnl = value - fee - entry;
          cash += value - fee;
          trades.push({{date: rows[i].date, type: "sell", close, price, shares, pnl}});
          pnls.push(pnl);
          shares = 0;
          entry = 0;
        }}
        const total = cash + shares * close;
        equity.push({{...rows[i], shortMa: shortMa[i], longMa: longMa[i], nav: total / capital, bench: close / rows[0].close, signal: golden ? 1 : death ? -1 : 0}});
      }}
      let peak = 1;
      const returns = [];
      equity.forEach((d, i) => {{
        peak = Math.max(peak, d.nav);
        d.drawdown = d.nav / peak - 1;
        returns.push(i === 0 ? 0 : d.nav / equity[i - 1].nav - 1);
      }});
      const totalReturn = equity[equity.length - 1].nav - 1;
      const annualReturn = Math.pow(1 + totalReturn, DATA.defaults.tradingDays / Math.max(equity.length, 1)) - 1;
      const dailyRf = Math.pow(1 + DATA.defaults.riskFreeRate, 1 / DATA.defaults.tradingDays) - 1;
      const mean = returns.reduce((a, b) => a + b - dailyRf, 0) / returns.length;
      const rawMean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const sd = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - rawMean, 2), 0) / Math.max(returns.length - 1, 1));
      const wins = pnls.filter(v => v > 0), losses = pnls.filter(v => v < 0);
      const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
      return {{
        equity, trades,
        totalReturn,
        annualReturn,
        sharpe: sd > 0 ? mean / sd * Math.sqrt(DATA.defaults.tradingDays) : 0,
        maxDrawdown: Math.min(...equity.map(d => d.drawdown)),
        winRate: pnls.length ? wins.length / pnls.length : 0,
        profitLossRatio: losses.length && wins.length ? avg(wins) / Math.abs(avg(losses)) : wins.length ? Infinity : 0,
        benchmarkReturn: equity[equity.length - 1].bench - 1,
      }};
    }}

    function scale(vals, a, b) {{
      const finite = vals.filter(v => Number.isFinite(v));
      let min = Math.min(...finite), max = Math.max(...finite);
      if (min === max) {{ min -= 1; max += 1; }}
      return v => a + (b - a) * (v - min) / (max - min);
    }}
    function line(points, x, y) {{
      return points.map((p, i) => (i ? "L" : "M") + x(p, i).toFixed(1) + "," + y(p).toFixed(1)).join(" ");
    }}
    function clear(svg) {{ svg.innerHTML = ""; }}
    function path(svg, d, color, width=2, dash="") {{
      svg.insertAdjacentHTML("beforeend", `<path d="${{d}}" fill="none" stroke="${{color}}" stroke-width="${{width}}" stroke-dasharray="${{dash}}"/>`);
    }}
    function axes(svg, w, h, m) {{
      svg.setAttribute("viewBox", `0 0 ${{w}} ${{h}}`);
      svg.insertAdjacentHTML("beforeend", `<rect x="${{m.l}}" y="${{m.t}}" width="${{w-m.l-m.r}}" height="${{h-m.t-m.b}}" fill="#fff"/><g stroke="#e8ebf2">${{[0,1,2,3,4].map(i=>`<line x1="${{m.l}}" x2="${{w-m.r}}" y1="${{m.t+i*(h-m.t-m.b)/4}}" y2="${{m.t+i*(h-m.t-m.b)/4}}"/>`).join("")}}</g>`);
    }}
    function drawNav(result) {{
      const svg = els.navChart, w = 900, h = 330, m = {{l:48,r:20,t:12,b:36}}; clear(svg); axes(svg,w,h,m);
      const points = result.equity, x = (_, i) => m.l + i * (w - m.l - m.r) / Math.max(points.length - 1, 1);
      const y = scale([...points.map(d=>d.nav), ...points.map(d=>d.bench)], h-m.b, m.t);
      path(svg, line(points, x, d=>y(d.nav)), "#d8343f", 2.4);
      path(svg, line(points, x, d=>y(d.bench)), "#9aa1ad", 2, "5 5");
      svg.insertAdjacentHTML("beforeend", `<text x="${{m.l}}" y="${{h-10}}" font-size="12" fill="#6b7280">${{points[0].date}}</text><text x="${{w-m.r-80}}" y="${{h-10}}" font-size="12" fill="#6b7280">${{points.at(-1).date}}</text>`);
    }}
    function drawDrawdown(result) {{
      const svg = els.drawdownChart, w = 540, h = 280, m = {{l:44,r:16,t:12,b:34}}; clear(svg); axes(svg,w,h,m);
      const points = result.equity, x = (_, i) => m.l + i * (w - m.l - m.r) / Math.max(points.length - 1, 1);
      const y = scale([...points.map(d=>d.drawdown * 100), 0], h-m.b, m.t);
      const area = line(points, x, d=>y(d.drawdown*100)) + ` L ${{w-m.r}},${{y(0)}} L ${{m.l}},${{y(0)}} Z`;
      svg.insertAdjacentHTML("beforeend", `<path d="${{area}}" fill="#cceee1"/><line x1="${{m.l}}" x2="${{w-m.r}}" y1="${{y(0)}}" y2="${{y(0)}}" stroke="#0c9f6e"/><path d="${{line(points,x,d=>y(d.drawdown*100))}}" fill="none" stroke="#0c9f6e" stroke-width="2"/>`);
    }}
    function drawPrice(result) {{
      const svg = els.priceChart, w = 540, h = 280, m = {{l:44,r:16,t:12,b:34}}; clear(svg); axes(svg,w,h,m);
      const points = result.equity, x = (_, i) => m.l + i * (w - m.l - m.r) / Math.max(points.length - 1, 1);
      const y = scale([...points.map(d=>d.close), ...points.map(d=>d.shortMa), ...points.map(d=>d.longMa)], h-m.b, m.t);
      path(svg, line(points, x, d=>y(d.close)), "#555", 1.4);
      path(svg, line(points.filter(d=>d.shortMa), (p,i)=>x(p, points.indexOf(p)), d=>y(d.shortMa)), "#f2a65a", 2);
      path(svg, line(points.filter(d=>d.longMa), (p,i)=>x(p, points.indexOf(p)), d=>y(d.longMa)), "#2f8ec4", 2);
      result.trades.forEach(t => {{
        const idx = points.findIndex(d => d.date === t.date);
        const cx = x(points[idx], idx), cy = y(points[idx].close), color = t.type === "buy" ? "#d8343f" : "#0c9f6e", marker = t.type === "buy" ? "▲" : "▼";
        svg.insertAdjacentHTML("beforeend", `<text x="${{cx-6}}" y="${{cy+4}}" fill="${{color}}" font-size="18">${{marker}}</text>`);
      }});
    }}
    function drawCompare() {{
      const svg = els.compareChart, w = 900, h = 330, m = {{l:48,r:20,t:12,b:86}}; clear(svg); axes(svg,w,h,m);
      const rows = DATA.comparison, xStep = (w-m.l-m.r)/rows.length, y = scale([...rows.map(d=>d.totalReturn*100), ...rows.map(d=>d.excessReturn*100), 0], h-m.b, m.t);
      rows.forEach((d,i) => {{
        const x = m.l + i*xStep + xStep*0.18, bw=xStep*0.26;
        [["totalReturn","#d8343f"],["excessReturn","#0c9f6e"]].forEach((pair,j)=> {{
          const v = d[pair[0]]*100, y0=y(0), yv=y(v), bh=Math.abs(y0-yv);
          svg.insertAdjacentHTML("beforeend", `<rect x="${{x+j*(bw+3)}}" y="${{Math.min(y0,yv)}}" width="${{bw}}" height="${{bh}}" fill="${{pair[1]}}"/><text x="${{x-6}}" y="${{h-54}}" font-size="10" fill="#5b6370" transform="rotate(-35 ${{x-6}},${{h-54}})">${{d.stock}} ${{d.short}}/${{d.long}}</text>`);
        }});
      }});
    }}

    function update() {{
      let shortW = +els.shortRange.value, longW = +els.longRange.value;
      if (longW <= shortW) {{ longW = shortW + 1; els.longRange.value = longW; }}
      els.shortValue.textContent = shortW; els.longValue.textContent = longW; els.paramPill.textContent = `SMA(${{shortW}},${{longW}})`;
      const s = stock();
      let rows = s.rows.filter(d => d.date >= els.startDate.value && d.date <= els.endDate.value);
      rows = rows.length > longW + 2 ? rows : s.rows;
      const result = runBacktest(rows, shortW, longW, +els.capitalInput.value, els.commissionToggle.checked, els.slippageToggle.checked);
      const mddDates = result.equity.filter(d => d.drawdown === result.maxDrawdown).map(d => d.date)[0] || "-";
      els.annualCard.textContent = pct(result.annualReturn); els.annualCard.className = "value " + cls(result.annualReturn);
      els.returnNote.textContent = "累计收益 " + pct(result.totalReturn) + "，超额 " + pct(result.totalReturn - result.benchmarkReturn);
      els.sharpeCard.textContent = num(result.sharpe, 3); els.sharpeCard.className = "value " + cls(result.sharpe);
      els.sharpeNote.textContent = result.sharpe > 1 ? "较好" : result.sharpe > 0 ? "中等" : "偏弱";
      els.mddCard.textContent = pct(result.maxDrawdown); els.mddNote.textContent = mddDates;
      els.winCard.textContent = pct(result.winRate); els.tradeNote.textContent = result.trades.length + " 笔交易，盈亏比 " + num(result.profitLossRatio, 2);
      els.sideAnnual.textContent = pct(result.annualReturn); els.sideSharpe.textContent = num(result.sharpe, 3); els.sideMdd.textContent = pct(result.maxDrawdown); els.sideTrades.textContent = result.trades.length; els.sideWin.textContent = pct(result.winRate); els.sidePl.textContent = num(result.profitLossRatio, 2); els.sideExcess.textContent = pct(result.totalReturn - result.benchmarkReturn);
      els.stockMeta.textContent = s.code + " · CNY · " + rows.length + " 天";
      drawNav(result); drawDrawdown(result); drawPrice(result); drawCompare();
    }}

    function init() {{
      DATA.stocks.forEach(s => els.stockSelect.insertAdjacentHTML("beforeend", `<option value="${{s.code}}">${{s.name}}（${{s.code}}）</option>`));
      els.stockSelect.value = DATA.defaults.stock;
      const s = stock();
      els.startDate.value = s.rows[0].date; els.endDate.value = s.rows.at(-1).date;
      els.shortRange.value = DATA.defaults.short; els.longRange.value = DATA.defaults.long; els.capitalInput.value = DATA.defaults.initialCapital;
      ["stockSelect","startDate","endDate","shortRange","longRange","commissionToggle","slippageToggle","capitalInput"].forEach(id => els[id].addEventListener("input", update));
      els.resetBtn.addEventListener("click", () => {{ els.stockSelect.value = DATA.defaults.stock; const s2 = stock(); els.startDate.value = s2.rows[0].date; els.endDate.value = s2.rows.at(-1).date; els.shortRange.value = DATA.defaults.short; els.longRange.value = DATA.defaults.long; els.capitalInput.value = DATA.defaults.initialCapital; els.commissionToggle.checked = true; els.slippageToggle.checked = true; update(); }});
      update();
    }}
    init();
  </script>
</body>
</html>
"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def build_readme(metrics_df: pd.DataFrame) -> None:
    main = metrics_df[
        (metrics_df["ts_code"] == MAIN_CODE)
        & (metrics_df["short_window"] == MAIN_PARAMS[0])
        & (metrics_df["long_window"] == MAIN_PARAMS[1])
    ].iloc[0]
    readme = f"""# TASK3 策略首秀：用均线交叉反应市场趋势变化

本目录完成双均线交叉策略作业，复用仓库中已经保存的 A 股日行情 CSV，生成交易信号、逐日回测、绩效指标、图表、Notebook 和 PDF 报告。

## 文件结构

```text
TASK3/
├── README.md
├── task3_ma_cross_strategy.py
├── task3_ma_cross_strategy.ipynb
├── ma_cross_dashboard.html
├── jane+TASK3.pdf
├── data/
│   └── *_daily_data.csv
├── figures/
│   └── figure*.png
└── results/
    ├── task3_performance_summary.csv
    ├── equity_*.csv
    └── trades_*.csv
```

## 运行方式

```bash
python3 task3_ma_cross_strategy.py
```

生成后可直接用浏览器打开 `ma_cross_dashboard.html`，查看双均线策略回测看板。看板包含标的选择、时间窗口、短长均线参数、交易成本开关、关键指标卡、策略净值曲线、回撤曲线、价格均线买卖点图，以及多股票多参数对比图。

脚本默认复用本地 CSV，不需要写入真实 Tushare token。若未来需要重新获取数据，请只通过本地环境变量传入：

```bash
export TUSHARE_TOKEN=\"YOUR_TUSHARE_TOKEN\"
```

## 主实验结果

- 标的：{main['stock_name']}（{MAIN_CODE}）
- 参数：MA{MAIN_PARAMS[0]}/MA{MAIN_PARAMS[1]}
- 初始资金：100000 元
- 手续费：0.03%
- 滑点：0.02%
- 累计回报：{pct(float(main['total_return']))}
- 年化收益：{pct(float(main['annual_return']))}
- 最大回撤：{pct(float(main['max_drawdown']))}
- 夏普比率：{float(main['sharpe']):.2f}
- 超额收益：{pct(float(main['excess_return']))}

## 说明

本项目仅用于课程学习和量化策略入门练习，不构成投资建议。仓库中不包含真实 Tushare token、完整 MCP URL、`.env` 文件、本机绝对路径或个人本机用户名。
"""
    README_PATH.write_text(readme, encoding="utf-8")


def build_outputs() -> None:
    data = load_all_data()
    metrics_df, equity_map, trade_map = make_all_backtests(data)
    make_figures(equity_map, trade_map, metrics_df)
    build_dashboard(data, metrics_df)
    build_pdf(metrics_df, equity_map, trade_map, data)
    build_notebook()
    build_readme(metrics_df)
    print(f"PDF: {PDF_PATH.name}")
    print(f"Dashboard: {DASHBOARD_PATH.name}")
    print(f"Summary: {RESULT_DIR / 'task3_performance_summary.csv'}")
    print(metrics_df[["stock_name", "ts_code", "short_window", "long_window", "total_return", "sharpe", "max_drawdown", "excess_return"]].to_string(index=False))


if __name__ == "__main__":
    build_outputs()
