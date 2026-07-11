#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK4 复刻传奇：海龟交易法则实战演练

本脚本复用仓库中已保存的 A 股日行情 CSV，完成海龟交易策略的指标计算、
逐日回测、参数对比、图表、Notebook、README 和 PDF 报告生成。

安全说明：
- 不写入真实 Tushare token、MCP URL、.env 内容或本机绝对路径。
- 如需重新获取行情，请在本地设置环境变量 TUSHARE_TOKEN；本脚本默认复用
  已保存 CSV，保证作业结果可复现。
"""

from __future__ import annotations

import json
import math
import shutil
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
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
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
SOURCE_DATA_DIR = PROJECT_DIR / "TASK3" / "data"
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

README_PATH = BASE_DIR / "README.md"
NB_PATH = BASE_DIR / "task4_turtle_strategy.ipynb"
PDF_PATH = OUTPUT_DIR / "jane+TASK4.pdf"
METRICS_PATH = OUTPUT_DIR / "metrics.csv"
TRADES_PATH = OUTPUT_DIR / "trades.csv"
PARAM_PATH = OUTPUT_DIR / "parameter_comparison.csv"

INITIAL_CASH = 100000.0
RISK_PER_UNIT = 0.01
MAX_UNITS = 4
TRADING_DAYS = 252

MAIN_CODE = "300750.SZ"
MAIN_NAME = "宁德时代"
MAIN_PARAMS = {"entry_window": 20, "exit_window": 10, "atr_window": 14}
PARAM_SETS = [
    {"entry_window": 20, "exit_window": 10, "atr_window": 14},
    {"entry_window": 55, "exit_window": 20, "atr_window": 14},
    {"entry_window": 20, "exit_window": 20, "atr_window": 14},
    {"entry_window": 20, "exit_window": 10, "atr_window": 20},
]

STOCKS = [
    {"name": "贵州茅台", "code": "600519.SH", "source_csv": "600519_SH_daily_data.csv"},
    {"name": "宁德时代", "code": "300750.SZ", "source_csv": "300750_SZ_daily_data.csv"},
    {"name": "招商银行", "code": "600036.SH", "source_csv": "600036_SH_daily_data.csv"},
]

REQUIRED_COLUMNS = ["trade_date", "open", "high", "low", "close", "vol"]


def choose_chinese_font() -> tuple[str, str | None]:
    candidates = [
        ("Songti SC", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
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
    for folder in [DATA_DIR, OUTPUT_DIR, FIG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {"volume": "vol", "date": "trade_date"}
    df = df.rename(columns=rename_map).copy()
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"行情数据缺少字段：{missing}")
    df = df.copy()
    date_text = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df["trade_date"] = pd.to_datetime(date_text, format="%Y%m%d", errors="coerce")
    for col in ["open", "high", "low", "close", "vol"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("trade_date").dropna(subset=["trade_date"]).reset_index(drop=True)


def load_all_data() -> dict[str, pd.DataFrame]:
    ensure_dirs()
    loaded: dict[str, pd.DataFrame] = {}
    for stock in STOCKS:
        source = SOURCE_DATA_DIR / stock["source_csv"]
        target = DATA_DIR / stock["source_csv"]
        if not source.exists():
            raise FileNotFoundError(f"缺少本地行情 CSV：{source}")
        shutil.copy2(source, target)
        df = normalize_daily_df(pd.read_csv(target))
        df["stock_name"] = stock["name"]
        df["ts_code"] = stock["code"]
        df.to_csv(target, index=False, encoding="utf-8-sig")
        loaded[stock["code"]] = df
    return loaded


def quality_checks(df: pd.DataFrame) -> dict[str, str]:
    price_cols = ["open", "high", "low", "close"]
    missing = int(df[REQUIRED_COLUMNS].isna().sum().sum())
    duplicate_dates = int(df["trade_date"].duplicated().sum())
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
        "样本数": f"{len(df)}",
        "日期范围": f"{df['trade_date'].min().date()} 至 {df['trade_date'].max().date()}",
        "缺失值": str(missing),
        "重复日期": str(duplicate_dates),
        "日期升序": "是" if df["trade_date"].is_monotonic_increasing else "否",
        "非正价格": str(int((df[price_cols] <= 0).sum().sum())),
        "负成交量": str(int((df["vol"] < 0).sum())),
        "OHLC 异常": str(ohlc_bad),
    }


def add_turtle_indicators(df: pd.DataFrame, entry_window: int, exit_window: int, atr_window: int) -> pd.DataFrame:
    out = df.sort_values("trade_date").reset_index(drop=True).copy()
    out["donchian_high"] = out["high"].rolling(entry_window).max().shift(1)
    out["donchian_low"] = out["low"].rolling(exit_window).min().shift(1)
    prev_close = out["close"].shift(1)
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - prev_close).abs()
    tr3 = (out["low"] - prev_close).abs()
    out["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    out["atr"] = out["tr"].rolling(atr_window).mean()
    out["buy_signal"] = out["close"] > out["donchian_high"]
    out["exit_signal"] = out["close"] < out["donchian_low"]
    return out


def run_turtle_backtest(
    df: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    atr_window: int = 14,
    initial_cash: float = INITIAL_CASH,
    risk_per_unit: float = RISK_PER_UNIT,
    max_units: int = MAX_UNITS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str], pd.DataFrame]:
    work = add_turtle_indicators(df, entry_window, exit_window, atr_window)
    cash = initial_cash
    units: list[dict[str, float]] = []
    next_add_price = np.nan
    trade_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = []

    for _, row in work.iterrows():
        date = row["trade_date"]
        close = float(row["close"])
        atr = float(row["atr"]) if pd.notna(row["atr"]) else np.nan
        action = ""
        trade_shares = 0
        trade_value = 0.0
        stop_price = max([u["stop_price"] for u in units], default=np.nan)

        exit_by_channel = bool(row["exit_signal"]) and units
        exit_by_stop = bool(units) and pd.notna(stop_price) and close <= stop_price
        if exit_by_channel or exit_by_stop:
            trade_shares = int(sum(u["shares"] for u in units))
            trade_value = trade_shares * close
            cash += trade_value
            reason = "跌破离场通道" if exit_by_channel else "触发2ATR止损"
            trade_rows.append(
                {
                    "trade_date": date,
                    "action": "SELL",
                    "reason": reason,
                    "price": close,
                    "shares": trade_shares,
                    "units": len(units),
                    "cash_after": cash,
                    "stop_price": stop_price,
                    "atr": atr,
                }
            )
            units = []
            next_add_price = np.nan
            action = "SELL"
        elif pd.notna(atr) and atr > 0:
            should_enter = (not units) and bool(row["buy_signal"])
            should_add = bool(units) and len(units) < max_units and pd.notna(next_add_price) and close >= next_add_price
            if should_enter or should_add:
                account_value = cash + sum(u["shares"] for u in units) * close
                unit_size = max(1, math.floor((account_value * risk_per_unit) / atr))
                affordable = math.floor(cash / close)
                shares = min(unit_size, affordable)
                if shares > 0:
                    cash -= shares * close
                    stop_for_unit = close - 2 * atr
                    units.append({"shares": float(shares), "entry_price": close, "atr": atr, "stop_price": stop_for_unit})
                    next_add_price = close + 0.5 * atr
                    trade_rows.append(
                        {
                            "trade_date": date,
                            "action": "BUY" if should_enter else "ADD",
                            "reason": "20日高点突破" if should_enter else "上涨0.5ATR金字塔加仓",
                            "price": close,
                            "shares": shares,
                            "units": len(units),
                            "cash_after": cash,
                            "stop_price": stop_for_unit,
                            "atr": atr,
                        }
                    )
                    action = "BUY" if should_enter else "ADD"

        total_shares = int(sum(u["shares"] for u in units))
        position_value = total_shares * close
        total_asset = cash + position_value
        equity_rows.append(
            {
                "trade_date": date,
                "close": close,
                "cash": cash,
                "shares": total_shares,
                "units": len(units),
                "position_value": position_value,
                "total_asset": total_asset,
                "action": action,
                "donchian_high": row["donchian_high"],
                "donchian_low": row["donchian_low"],
                "atr": row["atr"],
                "stop_price": max([u["stop_price"] for u in units], default=np.nan),
            }
        )

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(equity, trades, initial_cash)
    metrics.update({"entry_window": entry_window, "exit_window": exit_window, "atr_window": atr_window})
    return work, equity, metrics, trades


def calculate_metrics(equity: pd.DataFrame, trades: pd.DataFrame, initial_cash: float) -> dict[str, float | int | str]:
    nav = equity["total_asset"] / initial_cash
    daily_return = equity["total_asset"].pct_change().fillna(0)
    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    cumulative_return = float(nav.iloc[-1] - 1)
    annual_return = float(nav.iloc[-1] ** (TRADING_DAYS / max(len(nav), 1)) - 1)
    annual_vol = float(daily_return.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = float(daily_return.mean() / daily_return.std(ddof=0) * np.sqrt(TRADING_DAYS)) if daily_return.std(ddof=0) else 0.0
    sell_trades = trades[trades["action"] == "SELL"].copy() if not trades.empty else pd.DataFrame()
    wins = 0
    completed = 0
    if not trades.empty:
        cost = 0.0
        for _, row in trades.iterrows():
            if row["action"] in ("BUY", "ADD"):
                cost += float(row["price"]) * float(row["shares"])
            elif row["action"] == "SELL":
                completed += 1
                proceeds = float(row["price"]) * float(row["shares"])
                if proceeds > cost:
                    wins += 1
                cost = 0.0
    benchmark_return = float(equity["close"].iloc[-1] / equity["close"].iloc[0] - 1)
    return {
        "final_asset": float(equity["total_asset"].iloc[-1]),
        "cumulative_return": cumulative_return,
        "annual_return": annual_return,
        "max_drawdown": float(drawdown.min()),
        "sharpe_ratio": sharpe,
        "annual_volatility": annual_vol,
        "trade_count": int(len(trades)),
        "completed_round_trips": int(completed),
        "win_rate": float(wins / completed) if completed else 0.0,
        "benchmark_return": benchmark_return,
        "excess_return": cumulative_return - benchmark_return,
        "sell_count": int(len(sell_trades)),
    }


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def money(x: float) -> str:
    return f"{x:,.2f}"


def make_figures(main_df: pd.DataFrame, work: pd.DataFrame, equity: pd.DataFrame, trades: pd.DataFrame, param_metrics: pd.DataFrame) -> dict[str, Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "price_channel": FIG_DIR / "figure1_price_donchian_signals.png",
        "atr": FIG_DIR / "figure2_atr_volatility.png",
        "equity": FIG_DIR / "figure3_strategy_equity.png",
        "drawdown": FIG_DIR / "figure4_drawdown_curve.png",
        "params": FIG_DIR / "figure5_parameter_comparison.png",
    }

    buy_dates = trades.loc[trades["action"].isin(["BUY", "ADD"]), "trade_date"] if not trades.empty else []
    sell_dates = trades.loc[trades["action"] == "SELL", "trade_date"] if not trades.empty else []
    buy_points = work[work["trade_date"].isin(buy_dates)]
    sell_points = work[work["trade_date"].isin(sell_dates)]

    plt.figure(figsize=(11, 6))
    plt.plot(work["trade_date"], work["close"], label="收盘价", color="#2f5597", linewidth=1.5)
    plt.plot(work["trade_date"], work["donchian_high"], label="唐奇安上轨", color="#c55a11", linewidth=1.1)
    plt.plot(work["trade_date"], work["donchian_low"], label="唐奇安下轨", color="#70ad47", linewidth=1.1)
    plt.scatter(buy_points["trade_date"], buy_points["close"], marker="^", color="#d62728", s=55, label="买入/加仓")
    plt.scatter(sell_points["trade_date"], sell_points["close"], marker="v", color="#1f7a8c", s=55, label="卖出")
    plt.title("图1 宁德时代收盘价、唐奇安通道与交易信号")
    plt.xlabel("日期")
    plt.ylabel("价格")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figs["price_channel"], dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    plt.plot(work["trade_date"], work["atr"], color="#c00000", label="ATR")
    plt.title("图2 ATR 动态波动幅度")
    plt.xlabel("日期")
    plt.ylabel("ATR")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figs["atr"], dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    nav = equity["total_asset"] / INITIAL_CASH
    benchmark = main_df["close"] / main_df["close"].iloc[0]
    plt.plot(equity["trade_date"], nav, label="海龟策略净值", color="#2f5597", linewidth=1.6)
    plt.plot(main_df["trade_date"], benchmark, label="买入持有净值", color="#a5a5a5", linewidth=1.3)
    plt.title("图3 策略净值与买入持有对比")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figs["equity"], dpi=180)
    plt.close()

    plt.figure(figsize=(11, 4.8))
    drawdown = nav / nav.cummax() - 1
    plt.fill_between(equity["trade_date"], drawdown, 0, color="#9e2f2f", alpha=0.35)
    plt.plot(equity["trade_date"], drawdown, color="#9e2f2f", label="回撤")
    plt.title("图4 策略回撤曲线")
    plt.xlabel("日期")
    plt.ylabel("回撤")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figs["drawdown"], dpi=180)
    plt.close()

    labels = param_metrics["参数组合"]
    x = np.arange(len(labels))
    width = 0.25
    plt.figure(figsize=(12, 5.6))
    plt.bar(x - width, param_metrics["cumulative_return"], width, label="累计回报", color="#4472c4")
    plt.bar(x, param_metrics["max_drawdown"], width, label="最大回撤", color="#ed7d31")
    plt.bar(x + width, param_metrics["sharpe_ratio"], width, label="夏普比率", color="#70ad47")
    plt.axhline(0, color="#555555", linewidth=0.8)
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.title("图5 不同通道与 ATR 参数下的绩效对比")
    plt.ylabel("指标值")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figs["params"], dpi=180)
    plt.close()
    return figs


def build_parameter_experiment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for params in PARAM_SETS:
        _, _, metrics, _ = run_turtle_backtest(df, **params)
        label = f"E{params['entry_window']}/X{params['exit_window']}/ATR{params['atr_window']}"
        rows.append({"参数组合": label, **metrics})
    out = pd.DataFrame(rows)
    out.to_csv(PARAM_PATH, index=False, encoding="utf-8-sig")
    return out


def table_data_from_dict(d: dict[str, str]) -> list[list[str]]:
    return [["项目", "结果"], *[[k, v] for k, v in d.items()]]


def style_table(table: Table, header_color: colors.Color = colors.HexColor("#d9eaf7")) -> Table:
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), PDF_FONT),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("BACKGROUND", (0, 0), (-1, 0), header_color),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9e9e9e")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


PDF_FONT = register_pdf_font()


def make_pdf(
    data_quality: dict[str, str],
    metrics: dict[str, float | int | str],
    trades: pd.DataFrame,
    param_metrics: pd.DataFrame,
    figs: dict[str, Path],
) -> None:
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "ChineseNormal",
        parent=styles["Normal"],
        fontName=PDF_FONT,
        fontSize=10.5,
        leading=15.75,
        firstLineIndent=21,
        alignment=TA_JUSTIFY,
        spaceBefore=0,
        spaceAfter=0,
    )
    title = ParagraphStyle("Title", parent=normal, fontSize=18, leading=24, alignment=TA_CENTER, firstLineIndent=0)
    h1 = ParagraphStyle("H1", parent=normal, fontSize=13, leading=18, textColor=colors.HexColor("#c55a11"), firstLineIndent=0)
    caption = ParagraphStyle("Caption", parent=normal, fontSize=9.5, leading=13.5, alignment=TA_CENTER, firstLineIndent=0)
    code = ParagraphStyle("Code", parent=normal, fontName=PDF_FONT, fontSize=8.5, leading=11, leftIndent=12, firstLineIndent=0)

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=2.2 * cm,
        leftMargin=2.2 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="TASK4 海龟交易法则实战演练",
    )
    story: list[object] = []

    def p(text: str) -> None:
        story.append(Paragraph(text, normal))
        story.append(Spacer(1, 0.08 * cm))

    def section(text: str) -> None:
        story.append(Spacer(1, 0.15 * cm))
        story.append(Paragraph(text, h1))
        story.append(Spacer(1, 0.08 * cm))

    def add_image(path: Path, cap: str) -> None:
        story.append(Image(str(path), width=16.2 * cm, height=8.8 * cm if "figure1" in path.name else 7.4 * cm))
        story.append(Paragraph(cap, caption))
        story.append(Spacer(1, 0.12 * cm))

    story.append(Paragraph("复刻传奇：海龟交易法则实战演练", title))
    story.append(Spacer(1, 0.25 * cm))
    story.append(Paragraph("TASK4 量化交易策略回测报告", ParagraphStyle("SubTitle", parent=normal, alignment=TA_CENTER, firstLineIndent=0)))
    story.append(Spacer(1, 0.3 * cm))
    p("本报告依据课堂截图中的海龟交易法则五大要素完成：市场选择、买卖规模、突破入场、止损离场、风险控制，并使用 Python 对本地保存的 A 股日行情进行逐日回测。")

    section("一、任务背景与策略简介")
    p("海龟交易策略是经典趋势跟随策略。它不试图预测价格，而是通过价格突破过去一段时间的高点识别趋势启动，通过跌破过去一段时间的低点或触发 ATR 止损识别趋势结束。")
    p("本次实验选择宁德时代作为主回测标的，同时复用前序任务中的贵州茅台、招商银行数据作为可扩展样本。数据来源为前序任务已保存的 Tushare Pro 日线行情 CSV，代码默认读取本地文件，不展示或保存任何 token。")

    section("二、核心思想、优势与局限")
    p("海龟策略的核心思想是规则化地跟随趋势：突破买入、盈利后分批加仓、反向突破或动态止损离场，并用固定风险单位控制每次交易的最坏损失。")
    p("关键优势包括规则清晰、便于程序化执行、减少情绪干扰、在趋势行情中能捕捉较大收益，以及 ATR 可以适配不同波动环境。局限性也很明显：震荡行情中容易发生假突破，可能频繁止损，参数选择会显著影响收益和交易频率。")

    section("三、五大核心要素")
    p("1. 市场选择：本作业主标的为宁德时代，样本期为 2025 年 7 月至 2026 年 7 月，具有较明显的阶段性趋势和波动，适合观察趋势策略表现。")
    p("2. 仓位管理：课堂公式为可买股数 = 账户资金 × 1% / ATR。本实现使用账户总资产和 ATR 计算 Unit，并限制单市场最多 4 个单位。ATR 越大，Unit 越小；ATR 越小，Unit 越大。")
    p("3. 入场规则：使用唐奇安通道突破，价格突破过去 N 日最高价时买入。代码计算通道时使用 shift(1)，避免当天最高价参与当天信号。")
    p("4. 止损规则：使用 ATR 动态止损，止损价 = 入场价 - 2 × ATR。波动大时止损距离放宽，波动小时止损距离收紧。")
    p("5. 离场与止盈：策略不设置固定止盈，而是让利润奔跑；当价格跌破过去 10 日低点或触发 2ATR 止损时退出。")
    p("金字塔加仓规则为首次入场 1 个单位，价格每上涨 0.5 × ATR 增加 1 个单位，最多持有 4 个单位，并且不在亏损时加仓。")
    p("风险控制方面，单个市场最多 4 个单位，单单位风险约 1%，因此单市场理论风险约 4%；触发止损后立即卖出，不进行主观延迟。")

    section("四、数据预处理")
    p("程序统一字段名为 trade_date、open、high、low、close、vol，并按交易日期升序排列。数据检查包括缺失值、重复日期、非正价格、成交量和 OHLC 逻辑。")
    story.append(style_table(Table(table_data_from_dict(data_quality), colWidths=[5 * cm, 9 * cm])))
    story.append(Spacer(1, 0.18 * cm))

    section("五、Python 实现关键逻辑")
    snippet = """
df["donchian_high"] = df["high"].rolling(entry_window).max().shift(1)
df["donchian_low"] = df["low"].rolling(exit_window).min().shift(1)
prev_close = df["close"].shift(1)
tr1 = df["high"] - df["low"]
tr2 = (df["high"] - prev_close).abs()
tr3 = (df["low"] - prev_close).abs()
df["tr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df["atr"] = df["tr"].rolling(atr_window).mean()
df["buy_signal"] = df["close"] > df["donchian_high"]
df["exit_signal"] = df["close"] < df["donchian_low"]
stop_price = entry_price - 2 * atr_at_entry
"""
    story.append(Paragraph(snippet.replace("\n", "<br/>"), code))
    p("交易执行假设为：当日收盘价产生信号，并以当日收盘价成交；回测按交易日逐日推进，每一天只使用当天和以前的数据。")

    section("六、交易信号与可视化分析")
    add_image(figs["price_channel"], "图1 宁德时代收盘价、唐奇安通道与交易信号。买入点来自突破过去 20 日高点，加仓点来自上涨 0.5ATR，卖出点来自跌破离场通道或触发止损。")
    add_image(figs["atr"], "图2 ATR 动态波动幅度。ATR 衡量真实波动范围，决定止损距离和单位仓位大小。")

    section("七、策略回测结果")
    metric_rows = [
        ["指标", "数值"],
        ["最终资产", money(float(metrics["final_asset"]))],
        ["累计回报", pct(float(metrics["cumulative_return"]))],
        ["最大回撤", pct(float(metrics["max_drawdown"]))],
        ["夏普比率", f"{float(metrics['sharpe_ratio']):.2f}"],
        ["交易次数", str(int(metrics["trade_count"]))],
        ["完成交易轮数", str(int(metrics["completed_round_trips"]))],
        ["胜率", pct(float(metrics["win_rate"]))],
        ["买入持有收益", pct(float(metrics["benchmark_return"]))],
        ["超额收益", pct(float(metrics["excess_return"]))],
    ]
    story.append(style_table(Table(metric_rows, colWidths=[6 * cm, 6 * cm])))
    story.append(Spacer(1, 0.18 * cm))
    p("累计回报 = 最终资产 / 初始资产 - 1。最大回撤 = 当前净值 / 历史最高净值 - 1 的最小值，用来衡量账户从高点到低点的最大损失。夏普比率按日收益均值除以日收益标准差再乘以 sqrt(252) 计算，未扣除无风险利率。")
    add_image(figs["equity"], "图3 策略净值与买入持有对比。该图用于观察趋势跟随策略是否在样本期跑赢简单持有。")
    add_image(figs["drawdown"], "图4 策略回撤曲线。回撤越深，说明策略在不利行情中的资金压力越大。")

    section("八、参数对比实验")
    param_table = [["参数组合", "累计回报", "最大回撤", "夏普比率", "交易次数"]]
    for _, row in param_metrics.iterrows():
        param_table.append(
            [
                str(row["参数组合"]),
                pct(float(row["cumulative_return"])),
                pct(float(row["max_drawdown"])),
                f"{float(row['sharpe_ratio']):.2f}",
                str(int(row["trade_count"])),
            ]
        )
    story.append(style_table(Table(param_table, colWidths=[4.5 * cm, 3 * cm, 3 * cm, 3 * cm, 2.5 * cm])))
    story.append(Spacer(1, 0.18 * cm))
    add_image(figs["params"], "图5 不同通道与 ATR 参数下的绩效对比。20 日突破更敏感，55 日突破更稳健但交易机会较少；离场窗口和 ATR 窗口会影响退出速度与仓位大小。")

    section("九、适用场景、局限性与心得")
    p("海龟策略更适合趋势明确、波动延续性较强的市场。在单边上涨或下跌阶段，它可以通过突破入场和金字塔加仓扩大盈利；在横盘震荡中，则容易被假突破反复消耗。")
    p("本次实验最大的体会是：海龟法则并不是单纯的买卖信号，而是由入场、仓位、止损、加仓和风险上限共同组成的完整系统。只有把风险控制写进代码，趋势跟随策略才具备可执行性。")

    section("十、总结")
    p("本作业完成了海龟策略的理论说明、指标计算、逐日回测、交易记录、绩效评价、参数实验和图表报告。实现过程中严格使用本地公开 CSV 数据，不保存真实 token、私密 MCP URL 或本机绝对路径。")

    doc.build(story)


def make_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# TASK4 复刻传奇：海龟交易法则实战演练\n\n本 Notebook 调用同目录脚本完成海龟策略回测、图表和报告生成。"),
        nbf.v4.new_code_cell("from task4_turtle_strategy import main\nmain()"),
        nbf.v4.new_markdown_cell("运行后查看 `outputs/` 目录中的 PDF、交易记录、绩效指标和图表。"),
    ]
    NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")


def make_readme(metrics: dict[str, float | int | str]) -> None:
    README_PATH.write_text(
        f"""# TASK4 复刻传奇：海龟交易法则实战演练

本目录完成海龟交易策略的 Python 实现、逐日回测、参数实验、图表和 PDF 报告。

## 数据来源

默认复用仓库前序任务中已经保存的 A 股日行情 CSV，字段包括交易日期、开盘价、最高价、最低价、收盘价和成交量。报告中说明为 Tushare Pro 日线行情数据，但本目录不包含真实 token。

如需重新获取数据，请只通过环境变量设置 token：

```bash
export TUSHARE_TOKEN="YOUR_TUSHARE_TOKEN"
```

## 运行方法

```bash
python3 task4_turtle_strategy.py
```

## 主要参数

- `entry_window = 20`
- `exit_window = 10`
- `atr_window = 14`
- `initial_cash = 100000`
- `risk_per_unit = 0.01`
- `max_units = 4`

## 输出文件

- `outputs/jane+TASK4.pdf`：PDF 报告
- `outputs/metrics.csv`：主策略绩效指标
- `outputs/trades.csv`：交易记录
- `outputs/parameter_comparison.csv`：参数对比实验
- `outputs/figures/`：可视化图表
- `task4_turtle_strategy.ipynb`：Notebook 入口

## 主策略结果摘要

- 最终资产：{money(float(metrics['final_asset']))}
- 累计回报：{pct(float(metrics['cumulative_return']))}
- 最大回撤：{pct(float(metrics['max_drawdown']))}
- 夏普比率：{float(metrics['sharpe_ratio']):.2f}

## 安全说明

本目录不写入真实 Tushare token、MCP server URL、`.env` 内容、浏览器账号信息或本机绝对路径。
""",
        encoding="utf-8",
    )


def main() -> None:
    all_data = load_all_data()
    main_df = all_data[MAIN_CODE]
    work, equity, metrics, trades = run_turtle_backtest(main_df, **MAIN_PARAMS)
    param_metrics = build_parameter_experiment(main_df)
    figs = make_figures(main_df, work, equity, trades, param_metrics)

    metrics_df = pd.DataFrame([{**{"stock": MAIN_NAME, "ts_code": MAIN_CODE}, **metrics}])
    metrics_df.to_csv(METRICS_PATH, index=False, encoding="utf-8-sig")
    trades.to_csv(TRADES_PATH, index=False, encoding="utf-8-sig")
    equity.to_csv(OUTPUT_DIR / "equity_curve.csv", index=False, encoding="utf-8-sig")
    work.to_csv(OUTPUT_DIR / "indicator_data.csv", index=False, encoding="utf-8-sig")

    make_pdf(quality_checks(main_df), metrics, trades, param_metrics, figs)
    make_notebook()
    make_readme(metrics)

    print(json.dumps({"pdf": str(PDF_PATH), "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
