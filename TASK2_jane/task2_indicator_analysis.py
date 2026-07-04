#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK2 数据炼金术：数据诊断与构造交易指标

生成内容：
- 指标计算后的 CSV
- 五张静态 PNG 图表
- 可直接打开的交互式 HTML 指标看板
- Notebook
- PDF 报告
- README

说明：
- 主股票贵州茅台优先复用 TASK1 已保存 CSV。
- 宁德时代、招商银行作为网站对比股票，使用 AkShare 公开 A 股日行情接口补充。
- 所有路径均使用相对路径，不在输出文件中写入本机绝对路径。
"""

from __future__ import annotations

import json
import math
import shutil
import textwrap
import time
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
TASK1_CSV = PROJECT_DIR / "TASK1_jane_贵州茅台_600519" / "600519_SH_daily_data.csv"

MAIN_STOCK = {"name": "贵州茅台", "code": "600519.SH", "symbol": "600519", "csv": "600519_SH_daily_data.csv"}
COMPARE_STOCKS = [
    MAIN_STOCK,
    {"name": "宁德时代", "code": "300750.SZ", "symbol": "300750", "csv": "300750_SZ_daily_data.csv"},
    {"name": "招商银行", "code": "600036.SH", "symbol": "600036", "csv": "600036_SH_daily_data.csv"},
]

OUTPUT_CSV = BASE_DIR / "task2_600519_indicator_data.csv"
ALL_STOCKS_CSV = BASE_DIR / "task2_all_stocks_indicator_data.csv"
PDF_PATH = BASE_DIR / "jane+TASK2.pdf"
NB_PATH = BASE_DIR / "task2_indicator_analysis.ipynb"
HTML_PATH = BASE_DIR / "indicator_dashboard.html"
README_PATH = BASE_DIR / "README.md"

FIG_PRICE = BASE_DIR / "figure1_price_trend.png"
FIG_RSI = BASE_DIR / "figure2_rsi.png"
FIG_MACD = BASE_DIR / "figure3_macd.png"
FIG_BB = BASE_DIR / "figure4_bollinger_bands.png"
FIG_ATR = BASE_DIR / "figure5_atr_pct.png"
FIG_COMPARE = BASE_DIR / "figure6_cross_stock_comparison.png"
FIGURES = [FIG_PRICE, FIG_RSI, FIG_MACD, FIG_BB, FIG_ATR, FIG_COMPARE]

REQUIRED_COLUMNS = ["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]
DESC_COLUMNS = ["open", "high", "low", "close", "pct_chg", "vol", "amount"]


def choose_chinese_font() -> tuple[str, str | None]:
    candidates = [
        ("Songti SC", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("STSong", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
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


def copy_task1_main_csv() -> Path:
    if not TASK1_CSV.exists():
        raise FileNotFoundError("未找到 TASK1 中的 600519_SH_daily_data.csv，无法按要求复用本地数据。")
    target = BASE_DIR / MAIN_STOCK["csv"]
    if not target.exists():
        shutil.copy2(TASK1_CSV, target)
    return target


def normalize_daily_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, REQUIRED_COLUMNS].copy()
    date_text = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df["trade_date"] = pd.to_datetime(date_text, format="%Y%m%d")
    for col in REQUIRED_COLUMNS:
        if col != "trade_date":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("trade_date").reset_index(drop=True)


def fetch_akshare_stock(stock: dict[str, str], start_date: str, end_date: str) -> pd.DataFrame:
    import akshare as ak

    last_error: Exception | None = None
    raw = None
    for attempt in range(3):
        try:
            raw = ak.stock_zh_a_hist(
                symbol=stock["symbol"],
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
            break
        except Exception as exc:
            last_error = exc
            time.sleep(2 + attempt)
    if raw is None or raw.empty:
        raise RuntimeError(f"未能获取 {stock['name']}（{stock['code']}）公开日行情数据：{last_error}")

    df = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(raw["日期"]).dt.strftime("%Y%m%d"),
            "open": raw["开盘"],
            "high": raw["最高"],
            "low": raw["最低"],
            "close": raw["收盘"],
            "pre_close": raw["收盘"].shift(1),
            "change": raw["涨跌额"],
            "pct_chg": raw["涨跌幅"],
            "vol": raw["成交量"],
            "amount": raw["成交额"],
        }
    )
    df["pre_close"] = df["pre_close"].fillna(df["close"] - df["change"])
    return normalize_daily_df(df)


def load_stock_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    main_csv = copy_task1_main_csv()
    main_df = normalize_daily_df(pd.read_csv(main_csv))
    start_date = main_df["trade_date"].min().strftime("%Y%m%d")
    end_date = main_df["trade_date"].max().strftime("%Y%m%d")

    all_frames: list[pd.DataFrame] = []
    for stock in COMPARE_STOCKS:
        csv_path = BASE_DIR / stock["csv"]
        if stock["code"] == MAIN_STOCK["code"]:
            df = main_df.copy()
            source = "TASK1 本地 CSV"
        elif csv_path.exists():
            df = normalize_daily_df(pd.read_csv(csv_path))
            source = "AkShare 公开 A 股日行情接口（本地缓存）"
        else:
            df = fetch_akshare_stock(stock, start_date, end_date)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            source = "AkShare 公开 A 股日行情接口"
        df["stock_name"] = stock["name"]
        df["ts_code"] = stock["code"]
        df["data_source"] = source
        all_frames.append(df)

    all_df = pd.concat(all_frames, ignore_index=True)
    return main_df, all_df, "TASK1 本地 CSV；对比股票使用 AkShare 公开 A 股日行情接口"


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values("trade_date").copy()

    delta = out["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))
    out.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100
    out.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50

    out["ema_12"] = out["close"].ewm(span=12, adjust=False).mean()
    out["ema_26"] = out["close"].ewm(span=26, adjust=False).mean()
    out["dif"] = out["ema_12"] - out["ema_26"]
    out["dea"] = out["dif"].ewm(span=9, adjust=False).mean()
    out["macd_bar"] = 2 * (out["dif"] - out["dea"])

    out["bb_middle"] = out["close"].rolling(window=20, min_periods=20).mean()
    bb_std = out["close"].rolling(window=20, min_periods=20).std()
    out["bb_upper"] = out["bb_middle"] + 2 * bb_std
    out["bb_lower"] = out["bb_middle"] - 2 * bb_std
    out["bb_bandwidth"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]

    high_low = out["high"] - out["low"]
    high_prev_close = (out["high"] - out["close"].shift(1)).abs()
    low_prev_close = (out["low"] - out["close"].shift(1)).abs()
    out["tr"] = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    out["atr_14"] = out["tr"].rolling(window=14, min_periods=14).mean()
    out["atr_pct"] = out["atr_14"] / out["close"] * 100

    return out


def add_indicators_to_all(all_df: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _, group in all_df.groupby("ts_code", sort=False):
        extra_cols = group[["stock_name", "ts_code", "data_source"]].copy()
        base = group.drop(columns=["stock_name", "ts_code", "data_source"])
        calc = calculate_indicators(base)
        calc["stock_name"] = extra_cols["stock_name"].iloc[0]
        calc["ts_code"] = extra_cols["ts_code"].iloc[0]
        calc["data_source"] = extra_cols["data_source"].iloc[0]
        groups.append(calc)
    return pd.concat(groups, ignore_index=True)


def make_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    missing_by_col = df.isna().sum()
    duplicate_dates = int(df["trade_date"].duplicated().sum())
    sorted_ok = bool(df["trade_date"].is_monotonic_increasing)
    price_positive = bool((df[["open", "high", "low", "close"]] > 0).all().all())
    ohlc_ok = bool(
        (
            (df["high"] >= df["low"])
            & (df["high"] >= df["open"])
            & (df["high"] >= df["close"])
            & (df["low"] <= df["open"])
            & (df["low"] <= df["close"])
        ).all()
    )
    volume_ok = bool((df[["vol", "amount"]] >= 0).all().all())
    return pd.DataFrame(
        [
            ["缺失值检查", "对全部原始字段执行 isna().sum()", f"缺失值总数 {int(missing_by_col.sum())}", "通过" if int(missing_by_col.sum()) == 0 else "需核查"],
            ["重复交易日期", "检查 trade_date.duplicated()", f"重复日期 {duplicate_dates} 个", "通过" if duplicate_dates == 0 else "需核查"],
            ["日期升序", "检查 trade_date.is_monotonic_increasing", "已按升序排列" if sorted_ok else "未按升序排列", "通过" if sorted_ok else "需核查"],
            ["价格为正", "检查 open、high、low、close 是否均大于 0", "价格字段均为正" if price_positive else "存在非正价格", "通过" if price_positive else "需核查"],
            ["OHLC 逻辑", "检查 high >= low 且 high/low 能覆盖 open/close", "OHLC 关系合理" if ohlc_ok else "存在 OHLC 异常", "通过" if ohlc_ok else "需核查"],
            ["成交字段", "检查 vol、amount 是否非负", "成交量和成交额均非负" if volume_ok else "存在负数", "通过" if volume_ok else "需核查"],
        ],
        columns=["检查项目", "检查方法", "检查结果", "结论"],
    )


def make_desc_table(df: pd.DataFrame) -> pd.DataFrame:
    return df[DESC_COLUMNS].describe().T.round(3).reset_index().rename(columns={"index": "字段"})


def fmt_date(dt: pd.Timestamp) -> str:
    return pd.to_datetime(dt).strftime("%Y-%m-%d")


def summary_values(df: pd.DataFrame) -> dict[str, str | float]:
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    total_ret = (last_close / first_close - 1) * 100
    high_close = float(df["close"].max())
    low_close = float(df["close"].min())
    high_date = fmt_date(df.loc[df["close"].idxmax(), "trade_date"])
    low_date = fmt_date(df.loc[df["close"].idxmin(), "trade_date"])
    avg_pct = float(df["pct_chg"].mean())
    std_pct = float(df["pct_chg"].std())
    latest_rsi = float(df["rsi_14"].dropna().iloc[-1])
    latest_atr_pct = float(df["atr_pct"].dropna().iloc[-1])
    return {
        "start": fmt_date(df["trade_date"].min()),
        "end": fmt_date(df["trade_date"].max()),
        "rows": len(df),
        "first_close": first_close,
        "last_close": last_close,
        "total_ret": total_ret,
        "high_close": high_close,
        "low_close": low_close,
        "high_date": high_date,
        "low_date": low_date,
        "avg_pct": avg_pct,
        "std_pct": std_pct,
        "latest_rsi": latest_rsi,
        "latest_atr_pct": latest_atr_pct,
    }


def setup_axis(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel("交易日期", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def iter_stock_groups(all_data: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups: list[tuple[str, pd.DataFrame]] = []
    for stock in COMPARE_STOCKS:
        group = all_data[all_data["ts_code"] == stock["code"]].sort_values("trade_date")
        groups.append((f"{stock['name']}（{stock['code']}）", group))
    return groups


def plot_figures(all_data: pd.DataFrame) -> None:
    groups = iter_stock_groups(all_data)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=False)
    for ax, (label, df) in zip(axes, groups):
        ax.plot(df["trade_date"], df["close"], label="收盘价", color="#1f6f8b", linewidth=1.8)
        ax.plot(df["trade_date"], df["open"], label="开盘价", color="#8a5a00", linewidth=1.1, alpha=0.75)
        setup_axis(ax, f"{label} 收盘价走势与基础价格", "价格（元）")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("图1 三只股票收盘价走势与基础价格图", fontsize=15, y=0.995)
    save_fig(fig, FIG_PRICE)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=False)
    for ax, (label, df) in zip(axes, groups):
        ax.plot(df["trade_date"], df["rsi_14"], label="RSI(14)", color="#7b2cbf", linewidth=1.5)
        ax.axhline(70, color="#c0392b", linestyle="--", linewidth=0.9, label="70 阈值")
        ax.axhline(30, color="#117a65", linestyle="--", linewidth=0.9, label="30 阈值")
        ax.set_ylim(0, 100)
        setup_axis(ax, f"{label} RSI(14)", "RSI")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("图2 三只股票 RSI(14) 指标图", fontsize=15, y=0.995)
    save_fig(fig, FIG_RSI)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.2), sharex=False)
    for ax, (label, df) in zip(axes, groups):
        colors_bar = np.where(df["macd_bar"] >= 0, "#d1495b", "#2a9d8f")
        ax.bar(df["trade_date"], df["macd_bar"], label="MACD 柱", color=colors_bar, alpha=0.65, width=1.6)
        ax.plot(df["trade_date"], df["dif"], label="DIF", color="#214e9f", linewidth=1.2)
        ax.plot(df["trade_date"], df["dea"], label="DEA", color="#f28c28", linewidth=1.2)
        setup_axis(ax, f"{label} MACD(12,26,9)", "指标值")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("图3 三只股票 MACD(12,26,9) 指标图", fontsize=15, y=0.995)
    save_fig(fig, FIG_MACD)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.4), sharex=False)
    for ax, (label, df) in zip(axes, groups):
        ax.plot(df["trade_date"], df["close"], label="收盘价", color="#1f6f8b", linewidth=1.5)
        ax.plot(df["trade_date"], df["bb_middle"], label="中轨 MA20", color="#5f6877", linewidth=1.1)
        ax.plot(df["trade_date"], df["bb_upper"], label="上轨", color="#c0392b", linewidth=1.0)
        ax.plot(df["trade_date"], df["bb_lower"], label="下轨", color="#117a65", linewidth=1.0)
        ax.fill_between(
            mdates.date2num(df["trade_date"].to_numpy(dtype="datetime64[ns]")),
            df["bb_lower"].to_numpy(dtype=float),
            df["bb_upper"].to_numpy(dtype=float),
            color="#d9e8f5",
            alpha=0.32,
            label="布林带区间",
        )
        setup_axis(ax, f"{label} 布林带 Bollinger Bands(20,2)", "价格（元）")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("图4 三只股票布林带 Bollinger Bands(20,2) 图", fontsize=15, y=0.995)
    save_fig(fig, FIG_BB)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.5), sharex=False)
    for ax, (label, df) in zip(axes, groups):
        ax.plot(df["trade_date"], df["atr_pct"], label="ATR_PCT = ATR(14) / close", color="#b05d16", linewidth=1.5)
        setup_axis(ax, f"{label} ATR_PCT(14)", "ATR_PCT（%）")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("图5 三只股票 ATR_PCT(14) 相对波动率图", fontsize=15, y=0.995)
    save_fig(fig, FIG_ATR)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7.2), sharex=False)
    colors = ["#1f6f8b", "#c0392b", "#117a65"]
    for color, (label, df) in zip(colors, groups):
        norm_close = df["close"] / df["close"].iloc[0] * 100
        axes[0].plot(df["trade_date"], norm_close, label=label, color=color, linewidth=1.7)
        axes[1].plot(df["trade_date"], df["atr_pct"], label=label, color=color, linewidth=1.5)
    setup_axis(axes[0], "归一化收盘价走势（首日=100）", "指数化价格")
    setup_axis(axes[1], "ATR_PCT 相对波动率横向比较", "ATR_PCT（%）")
    axes[0].legend(loc="best", fontsize=8)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle("图6 三只股票横向对比图", fontsize=15, y=0.995)
    save_fig(fig, FIG_COMPARE)


def safe_num(value: float | int | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.4f}"


def make_dashboard(all_data: pd.DataFrame) -> None:
    records = {}
    for code, group in all_data.groupby("ts_code", sort=False):
        cols = [
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pct_chg",
            "rsi_14",
            "dif",
            "dea",
            "macd_bar",
            "bb_middle",
            "bb_upper",
            "bb_lower",
            "atr_14",
            "atr_pct",
            "stock_name",
            "ts_code",
        ]
        g = group[cols].copy()
        g["trade_date"] = g["trade_date"].dt.strftime("%Y-%m-%d")
        records[code] = g.where(pd.notna(g), None).to_dict(orient="records")
    data_json = json.dumps(records, ensure_ascii=False)
    stock_options = "\n".join(
        f'<option value="{stock["code"]}">{stock["name"]}（{stock["code"]}）</option>' for stock in COMPARE_STOCKS
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TASK2 交互式技术指标看板</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: #18212f;
      background: #f4f6f8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.55;
    }}
    main {{ width: min(1180px, calc(100% - 28px)); margin: 0 auto; padding: 22px 0 32px; }}
    header {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 16px;
      align-items: end;
      padding: 18px 0 16px;
      border-bottom: 1px solid #d7dde7;
    }}
    h1 {{ margin: 0; font-size: clamp(22px, 3vw, 34px); letter-spacing: 0; line-height: 1.2; }}
    .meta {{ margin-top: 6px; color: #5b6472; font-size: 14px; }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(210px, 1fr) auto auto;
      gap: 12px;
      align-items: center;
      padding: 14px 0;
    }}
    select, button {{
      min-height: 38px;
      border: 1px solid #c9d1dc;
      border-radius: 6px;
      background: #fff;
      color: #18212f;
      font-size: 14px;
      padding: 8px 10px;
    }}
    .ranges, .toggles {{ display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }}
    button {{ cursor: pointer; }}
    button.active {{ color: #fff; background: #2454a6; border-color: #2454a6; }}
    label.toggle {{
      display: inline-flex;
      align-items: center;
      gap: 5px;
      min-height: 38px;
      padding: 7px 10px;
      border: 1px solid #c9d1dc;
      border-radius: 6px;
      background: #fff;
      font-size: 14px;
      white-space: nowrap;
    }}
    .panel {{
      margin-top: 14px;
      padding: 16px;
      border: 1px solid #d7dde7;
      border-radius: 8px;
      background: #fff;
    }}
    .kpis {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
    .kpi {{ padding: 12px; border: 1px solid #e0e5ee; border-radius: 6px; background: #fbfcfe; }}
    .kpi span {{ display: block; color: #647084; font-size: 12px; }}
    .kpi strong {{ display: block; margin-top: 4px; font-size: 18px; overflow-wrap: anywhere; }}
    .charts {{ display: grid; gap: 14px; margin-top: 14px; }}
    .chartbox {{ padding: 12px; border: 1px solid #d7dde7; border-radius: 8px; background: #fff; }}
    canvas {{ display: block; width: 100%; height: 260px; }}
    #priceCanvas {{ height: 340px; }}
    .params {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
    .param {{ padding: 12px; border-left: 4px solid #2454a6; background: #f7f9fc; border-radius: 4px; }}
    .param strong {{ display: block; margin-bottom: 4px; }}
    .tooltip {{
      position: fixed;
      z-index: 20;
      min-width: 210px;
      max-width: 280px;
      display: none;
      padding: 10px;
      border: 1px solid #c9d1dc;
      border-radius: 6px;
      background: rgba(255,255,255,.96);
      box-shadow: 0 10px 28px rgba(28,38,54,.18);
      font-size: 12px;
      pointer-events: none;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 900px) {{
      header, .toolbar {{ grid-template-columns: 1fr; align-items: start; }}
      .ranges, .toggles {{ justify-content: flex-start; }}
      .kpis, .params {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 560px) {{
      main {{ width: calc(100% - 18px); }}
      .kpis, .params {{ grid-template-columns: 1fr; }}
      canvas, #priceCanvas {{ height: 260px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>TASK2 交互式技术指标看板</h1>
        <div class="meta">价格、布林带、RSI、MACD、ATR_PCT 可交互查看；悬停图表查看日期与指标值。</div>
      </div>
      <div class="meta">主股票：贵州茅台（600519.SH）</div>
    </header>

    <section class="toolbar">
      <select id="stockSelect" aria-label="股票选择">{stock_options}</select>
      <div class="ranges" aria-label="时间范围">
        <button data-range="all" class="active">全部</button>
        <button data-range="365">近 1 年</button>
        <button data-range="183">近 6 个月</button>
        <button data-range="92">近 3 个月</button>
      </div>
      <div class="toggles" aria-label="指标显示开关">
        <label class="toggle"><input type="checkbox" data-chart="bb" checked>布林带</label>
        <label class="toggle"><input type="checkbox" data-chart="rsi" checked>RSI</label>
        <label class="toggle"><input type="checkbox" data-chart="macd" checked>MACD</label>
        <label class="toggle"><input type="checkbox" data-chart="atr" checked>ATR</label>
      </div>
    </section>

    <section class="panel">
      <div class="kpis" id="kpis"></div>
    </section>

    <section class="charts">
      <div class="chartbox">
        <canvas id="priceCanvas" aria-label="价格和布林带图"></canvas>
      </div>
      <div class="chartbox" id="rsiBox">
        <canvas id="rsiCanvas" aria-label="RSI 指标图"></canvas>
      </div>
      <div class="chartbox" id="macdBox">
        <canvas id="macdCanvas" aria-label="MACD 指标图"></canvas>
      </div>
      <div class="chartbox" id="atrBox">
        <canvas id="atrCanvas" aria-label="ATR_PCT 波动率图"></canvas>
      </div>
    </section>

    <section class="panel">
      <div class="params">
        <div class="param"><strong>RSI(14)</strong>先计算涨跌幅，再计算 14 日平均上涨、平均下跌、RS，最后得到 RSI。</div>
        <div class="param"><strong>MACD(12,26,9)</strong>EMA12 减 EMA26 得 DIF，DIF 的 9 日 EMA 为 DEA，柱体为 2 × (DIF - DEA)。</div>
        <div class="param"><strong>布林带(20,2)</strong>中轨为 20 日均线，上下轨为中轨加减 2 倍标准差。</div>
        <div class="param"><strong>ATR(14)</strong>TR 取三项最大值，ATR 为 14 日 TR 均值；ATR_PCT 用于相对波动率比较。</div>
      </div>
    </section>
  </main>
  <div class="tooltip" id="tooltip"></div>

  <script>
    const DATA = {data_json};
    const state = {{ stock: "600519.SH", range: "all", bb: true, rsi: true, macd: true, atr: true }};
    const tooltip = document.getElementById("tooltip");
    const canvases = {{
      price: document.getElementById("priceCanvas"),
      rsi: document.getElementById("rsiCanvas"),
      macd: document.getElementById("macdCanvas"),
      atr: document.getElementById("atrCanvas")
    }};

    function fmt(v, digits = 2) {{
      return v === null || v === undefined || Number.isNaN(Number(v)) ? "-" : Number(v).toFixed(digits);
    }}
    function filteredData() {{
      let rows = DATA[state.stock] || [];
      if (state.range !== "all" && rows.length) {{
        const last = new Date(rows[rows.length - 1].trade_date);
        const cutoff = new Date(last);
        cutoff.setDate(cutoff.getDate() - Number(state.range));
        rows = rows.filter(r => new Date(r.trade_date) >= cutoff);
      }}
      return rows;
    }}
    function resizeCanvas(canvas) {{
      const ratio = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = Math.max(320, Math.floor(rect.width * ratio));
      canvas.height = Math.max(220, Math.floor(rect.height * ratio));
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {{ ctx, w: rect.width, h: rect.height }};
    }}
    function extent(rows, keys, pad = 0.05) {{
      const vals = [];
      rows.forEach(r => keys.forEach(k => {{
        if (r[k] !== null && r[k] !== undefined && Number.isFinite(Number(r[k]))) vals.push(Number(r[k]));
      }}));
      if (!vals.length) return [0, 1];
      let min = Math.min(...vals), max = Math.max(...vals);
      if (min === max) {{ min -= 1; max += 1; }}
      const p = (max - min) * pad;
      return [min - p, max + p];
    }}
    function drawFrame(ctx, w, h, title, yLabel) {{
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = "#18212f";
      ctx.font = "16px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillText(title, 14, 22);
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      ctx.fillStyle = "#647084";
      ctx.fillText(yLabel, 14, 42);
      const plot = {{ x: 54, y: 52, w: w - 74, h: h - 86 }};
      ctx.strokeStyle = "#d7dde7";
      ctx.lineWidth = 1;
      ctx.strokeRect(plot.x, plot.y, plot.w, plot.h);
      for (let i = 1; i < 5; i++) {{
        const y = plot.y + plot.h * i / 5;
        ctx.beginPath(); ctx.moveTo(plot.x, y); ctx.lineTo(plot.x + plot.w, y); ctx.stroke();
      }}
      return plot;
    }}
    function makeScale(rows, keys, plot, customExtent = null) {{
      const [minY, maxY] = customExtent || extent(rows, keys);
      const x = i => plot.x + (rows.length <= 1 ? 0 : i * plot.w / (rows.length - 1));
      const y = v => plot.y + plot.h - (Number(v) - minY) * plot.h / (maxY - minY);
      return {{ x, y, minY, maxY }};
    }}
    function drawLine(ctx, rows, key, scale, color, width = 1.6) {{
      ctx.strokeStyle = color; ctx.lineWidth = width; ctx.beginPath();
      let started = false;
      rows.forEach((r, i) => {{
        const v = r[key];
        if (v === null || v === undefined || !Number.isFinite(Number(v))) {{ started = false; return; }}
        const px = scale.x(i), py = scale.y(v);
        if (!started) {{ ctx.moveTo(px, py); started = true; }} else {{ ctx.lineTo(px, py); }}
      }});
      ctx.stroke();
    }}
    function drawLegend(ctx, items, x, y) {{
      ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
      let dx = 0;
      items.forEach(item => {{
        ctx.fillStyle = item.color; ctx.fillRect(x + dx, y - 9, 12, 8);
        ctx.fillStyle = "#303a49"; ctx.fillText(item.label, x + dx + 16, y);
        dx += ctx.measureText(item.label).width + 34;
      }});
    }}
    function drawPrice(rows) {{
      const {{ctx, w, h}} = resizeCanvas(canvases.price);
      const keys = state.bb ? ["close", "bb_middle", "bb_upper", "bb_lower"] : ["close"];
      const plot = drawFrame(ctx, w, h, state.bb ? "价格走势与布林带" : "价格走势", "价格（元）");
      const scale = makeScale(rows, keys, plot);
      if (state.bb) {{
        drawLine(ctx, rows, "bb_upper", scale, "#c0392b", 1.2);
        drawLine(ctx, rows, "bb_middle", scale, "#647084", 1.2);
        drawLine(ctx, rows, "bb_lower", scale, "#117a65", 1.2);
      }}
      drawLine(ctx, rows, "close", scale, "#1f6f8b", 1.9);
      drawLegend(ctx, state.bb ? [
        {{label: "收盘价", color: "#1f6f8b"}}, {{label: "中轨", color: "#647084"}},
        {{label: "上轨", color: "#c0392b"}}, {{label: "下轨", color: "#117a65"}}
      ] : [{{label: "收盘价", color: "#1f6f8b"}}], plot.x, h - 18);
    }}
    function drawRSI(rows) {{
      const box = document.getElementById("rsiBox");
      box.classList.toggle("hidden", !state.rsi);
      if (!state.rsi) return;
      const {{ctx, w, h}} = resizeCanvas(canvases.rsi);
      const plot = drawFrame(ctx, w, h, "RSI(14)", "RSI");
      const scale = makeScale(rows, ["rsi_14"], plot, [0, 100]);
      ctx.strokeStyle = "#c0392b"; ctx.setLineDash([6, 4]);
      [70, 30].forEach(v => {{ ctx.beginPath(); ctx.moveTo(plot.x, scale.y(v)); ctx.lineTo(plot.x + plot.w, scale.y(v)); ctx.stroke(); }});
      ctx.setLineDash([]);
      drawLine(ctx, rows, "rsi_14", scale, "#7b2cbf", 1.8);
      drawLegend(ctx, [{{label: "RSI(14)", color: "#7b2cbf"}}, {{label: "70/30 阈值", color: "#c0392b"}}], plot.x, h - 18);
    }}
    function drawMACD(rows) {{
      const box = document.getElementById("macdBox");
      box.classList.toggle("hidden", !state.macd);
      if (!state.macd) return;
      const {{ctx, w, h}} = resizeCanvas(canvases.macd);
      const plot = drawFrame(ctx, w, h, "MACD(12,26,9)", "指标值");
      const scale = makeScale(rows, ["dif", "dea", "macd_bar"], plot);
      const zero = scale.y(0);
      rows.forEach((r, i) => {{
        const v = Number(r.macd_bar);
        if (!Number.isFinite(v)) return;
        const x = scale.x(i), y = scale.y(v);
        ctx.fillStyle = v >= 0 ? "rgba(209,73,91,.65)" : "rgba(42,157,143,.65)";
        ctx.fillRect(x - 1.4, Math.min(y, zero), 2.8, Math.abs(zero - y));
      }});
      drawLine(ctx, rows, "dif", scale, "#214e9f", 1.5);
      drawLine(ctx, rows, "dea", scale, "#f28c28", 1.5);
      drawLegend(ctx, [{{label: "DIF", color: "#214e9f"}}, {{label: "DEA", color: "#f28c28"}}, {{label: "MACD 柱", color: "#d1495b"}}], plot.x, h - 18);
    }}
    function drawATR(rows) {{
      const box = document.getElementById("atrBox");
      box.classList.toggle("hidden", !state.atr);
      if (!state.atr) return;
      const {{ctx, w, h}} = resizeCanvas(canvases.atr);
      const plot = drawFrame(ctx, w, h, "ATR_PCT(14)", "相对波动率（%）");
      const scale = makeScale(rows, ["atr_pct"], plot);
      drawLine(ctx, rows, "atr_pct", scale, "#b05d16", 1.8);
      drawLegend(ctx, [{{label: "ATR_PCT", color: "#b05d16"}}], plot.x, h - 18);
    }}
    function updateKpis(rows) {{
      const first = rows[0], last = rows[rows.length - 1];
      const ret = (last.close / first.close - 1) * 100;
      document.getElementById("kpis").innerHTML = [
        ["股票", `${{last.stock_name}}（${{last.ts_code}}）`],
        ["时间范围", `${{first.trade_date}} 至 ${{last.trade_date}}`],
        ["末日收盘价", fmt(last.close)],
        ["区间涨跌幅", `${{fmt(ret)}}%`],
        ["最新 RSI / ATR_PCT", `${{fmt(last.rsi_14)}} / ${{fmt(last.atr_pct)}}%`],
      ].map(([k, v]) => `<div class="kpi"><span>${{k}}</span><strong>${{v}}</strong></div>`).join("");
    }}
    function render() {{
      const rows = filteredData();
      if (!rows.length) return;
      updateKpis(rows);
      drawPrice(rows); drawRSI(rows); drawMACD(rows); drawATR(rows);
    }}
    function nearestPoint(canvas, event) {{
      const rows = filteredData();
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const plotX = 54, plotW = rect.width - 74;
      const idx = Math.max(0, Math.min(rows.length - 1, Math.round((x - plotX) / plotW * (rows.length - 1))));
      return rows[idx];
    }}
    Object.values(canvases).forEach(canvas => {{
      canvas.addEventListener("mousemove", event => {{
        const r = nearestPoint(canvas, event);
        if (!r) return;
        tooltip.style.display = "block";
        tooltip.style.left = `${{event.clientX + 14}}px`;
        tooltip.style.top = `${{event.clientY + 14}}px`;
        tooltip.innerHTML = `<strong>${{r.trade_date}}</strong><br>收盘价：${{fmt(r.close)}}<br>RSI：${{fmt(r.rsi_14)}}<br>DIF/DEA：${{fmt(r.dif)}} / ${{fmt(r.dea)}}<br>MACD柱：${{fmt(r.macd_bar)}}<br>ATR_PCT：${{fmt(r.atr_pct)}}%`;
      }});
      canvas.addEventListener("mouseleave", () => tooltip.style.display = "none");
    }});
    document.getElementById("stockSelect").addEventListener("change", e => {{ state.stock = e.target.value; render(); }});
    document.querySelectorAll("[data-range]").forEach(btn => btn.addEventListener("click", () => {{
      document.querySelectorAll("[data-range]").forEach(b => b.classList.remove("active"));
      btn.classList.add("active"); state.range = btn.dataset.range; render();
    }}));
    document.querySelectorAll("[data-chart]").forEach(input => input.addEventListener("change", () => {{
      state[input.dataset.chart] = input.checked; render();
    }}));
    window.addEventListener("resize", render);
    render();
  </script>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def table_from_df(df: pd.DataFrame, style: ParagraphStyle, widths: list[float]) -> Table:
    rows = [[Paragraph(str(x), style) for x in df.columns]]
    for _, row in df.iterrows():
        rows.append([Paragraph(str(row[col]), style) for col in df.columns])
    table = Table(rows, colWidths=widths, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def add_heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.14 * cm))
    story.append(p(text, style))
    story.append(Spacer(1, 0.08 * cm))


def add_figure(
    story: list,
    path: Path,
    caption_text: str,
    caption: ParagraphStyle,
    width: float = 15.7 * cm,
    ratio: float = 0.64,
) -> None:
    story.append(KeepTogether([Image(str(path), width=width, height=width * ratio), p(caption_text, caption)]))
    story.append(Spacer(1, 0.12 * cm))


def make_all_quality_table(all_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in iter_stock_groups(all_data):
        raw = df[REQUIRED_COLUMNS].copy()
        missing_total = int(raw.isna().sum().sum())
        duplicate_dates = int(raw["trade_date"].duplicated().sum())
        sorted_ok = bool(raw["trade_date"].is_monotonic_increasing)
        price_positive = bool((raw[["open", "high", "low", "close"]] > 0).all().all())
        ohlc_ok = bool(
            (
                (raw["high"] >= raw["low"])
                & (raw["high"] >= raw["open"])
                & (raw["high"] >= raw["close"])
                & (raw["low"] <= raw["open"])
                & (raw["low"] <= raw["close"])
            ).all()
        )
        rows.append(
            {
                "股票": label,
                "时间范围": f"{fmt_date(raw['trade_date'].min())} 至 {fmt_date(raw['trade_date'].max())}",
                "行数": len(raw),
                "缺失": missing_total,
                "重复日期": duplicate_dates,
                "升序": "是" if sorted_ok else "否",
                "价格合理": "是" if price_positive and ohlc_ok else "否",
            }
        )
    return pd.DataFrame(rows)


def make_cross_summary_table(all_data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, df in iter_stock_groups(all_data):
        vals = summary_values(df)
        rows.append(
            {
                "股票": label,
                "区间涨跌幅": f"{vals['total_ret']:.2f}%",
                "日涨跌幅均值": f"{vals['avg_pct']:.3f}%",
                "日涨跌幅标准差": f"{vals['std_pct']:.3f}",
                "平均成交量": f"{df['vol'].mean():.0f}",
                "最新RSI": f"{vals['latest_rsi']:.2f}",
                "最新ATR_PCT": f"{vals['latest_atr_pct']:.2f}%",
            }
        )
    return pd.DataFrame(rows)


def make_report(all_data: pd.DataFrame, source_note: str) -> None:
    font_name = register_pdf_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "SongBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13.5,
        alignment=TA_JUSTIFY,
        firstLineIndent=18,
        spaceBefore=0,
        spaceAfter=0,
        wordWrap="CJK",
    )
    body_no_indent = ParagraphStyle(
        "SongBodyNoIndent",
        parent=body,
        firstLineIndent=0,
        alignment=TA_LEFT,
    )
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=body_no_indent,
        fontSize=10,
        leading=15,
        alignment=TA_CENTER,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=17,
        spaceBefore=3,
        spaceAfter=0,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body_no_indent,
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
    )
    small = ParagraphStyle(
        "Small",
        parent=body_no_indent,
        fontSize=7.6,
        leading=10.5,
    )

    main_df = all_data[all_data["ts_code"] == MAIN_STOCK["code"]].sort_values("trade_date")
    vals = summary_values(main_df)
    quality_df = make_all_quality_table(all_data)
    cross_df = make_cross_summary_table(all_data)
    source_rows = []
    for stock in COMPARE_STOCKS:
        group = all_data[all_data["ts_code"] == stock["code"]]
        source_rows.append(
            {
                "股票名称": stock["name"],
                "股票代码": stock["code"],
                "数据来源": str(group["data_source"].iloc[0]),
                "时间范围": f"{fmt_date(group['trade_date'].min())} 至 {fmt_date(group['trade_date'].max())}",
                "交易日数": len(group),
            }
        )
    source_df = pd.DataFrame(source_rows)

    story: list = [
        p("TASK2 数据炼金术：数据诊断与构造交易指标", title),
        p("姓名：jane<br/>分析对象：贵州茅台、宁德时代、招商银行", subtitle),
        Spacer(1, 0.3 * cm),
    ]

    add_heading(story, "一、任务背景", heading)
    story.append(p("在量化交易中，数据质量和指标构造会直接影响策略研究的可靠性。本报告分析贵州茅台（600519.SH）、宁德时代（300750.SZ）和招商银行（600036.SH）三只股票，完成数据诊断、描述性统计、RSI、MACD、布林带和 ATR 指标计算，并通过静态图表和交互式网页展示结果。报告仅用于课程学习，不构成投资建议。", body))

    add_heading(story, "二、数据说明与数据诊断", heading)
    story.append(p(f"数据来源：{source_note}。字段含义如下：trade_date 为交易日期，open/high/low/close 分别为开盘价、最高价、最低价、收盘价，pre_close 为前收盘价，change 为涨跌额，pct_chg 为涨跌幅，vol 为成交量，amount 为成交额。三只股票均采用同一时间窗口，便于横向比较。", body))
    story.append(table_from_df(source_df, small, [2.25 * cm, 2.2 * cm, 5.0 * cm, 3.4 * cm, 1.6 * cm]))
    story.append(Spacer(1, 0.12 * cm))
    story.append(table_from_df(quality_df, small, [3.3 * cm, 3.1 * cm, 1.0 * cm, 1.0 * cm, 1.3 * cm, 1.0 * cm, 1.5 * cm]))
    story.append(Spacer(1, 0.08 * cm))
    story.append(p("诊断结果显示，三只股票在本次样本中均不存在原始字段缺失值和重复交易日期，日期均已按升序排列，价格字段为正，OHLC 逻辑通过检查，成交量和成交额未出现负值。因此，三组数据可作为后续指标计算的基础。需要注意的是，数据诊断只能说明样本内部一致性较好，并不代表指标具有确定预测能力。", body))
    story.append(table_from_df(cross_df, small, [3.3 * cm, 1.9 * cm, 1.9 * cm, 2.1 * cm, 2.0 * cm, 1.6 * cm, 1.9 * cm]))
    story.append(p("从描述性统计看，宁德时代区间涨幅较高，收益波动也更明显；贵州茅台在样本区间内收盘价整体回落，末期 RSI 接近低位区域；招商银行价格水平较低但成交量较大，ATR_PCT 反映的相对波动可用于和另外两只股票同口径比较。", body))

    add_heading(story, "三、基础交易指标原理", heading)
    story.append(p("RSI（Relative Strength Index，相对强弱指标）用于衡量一段时间内平均上涨幅度与平均下跌幅度的相对关系。RSI(14) 的流程是先计算收盘价日变化，将上涨部分记为 gain、下跌绝对值记为 loss，再计算 14 日平均上涨和 14 日平均下跌，RS = 平均上涨 / 平均下跌，RSI = 100 - 100 / (1 + RS)。常用参数为 14，常见观察阈值为 70 和 30。RSI 可帮助观察价格是否处于相对强势或弱势区域，但在趋势行情中可能长期高位或低位，不能机械视为确定买卖信号。", body))
    story.append(p("MACD（Moving Average Convergence Divergence，指数平滑异同移动平均线）通过快慢指数移动平均线差值观察趋势和动能。常用 MACD(12,26,9)：EMA12 - EMA26 得 DIF，DIF 的 9 日 EMA 得 DEA，MACD 柱通常取 2 × (DIF - DEA)。DIF 和 DEA 的相对位置、柱体正负和扩大收缩可用于观察趋势动能变化。其局限是基于移动平均，天然滞后，在震荡市中容易出现频繁信号。", body))
    story.append(p("Bollinger Bands（布林带）用移动均线和标准差刻画价格的相对位置和波动范围。常用 Bollinger Bands(20,2)：中轨为 20 日移动平均线，上轨 = 中轨 + 2 × 20 日标准差，下轨 = 中轨 - 2 × 20 日标准差，带宽可用 (上轨 - 下轨) / 中轨表示。布林带可用于观察价格是否偏离近期均值和波动率是否扩张，但价格触及上轨或下轨并不自动代表反转，强趋势中价格可能沿轨运行。", body))

    add_heading(story, "四、Python 实现过程", heading)
    story.append(p("Python 脚本首先读取本地 CSV 并转换日期格式，然后按交易日期升序排列。RSI 使用 close.diff() 得到涨跌幅，分别计算 gain、loss、14 日平均上涨和平均下跌。MACD 使用 pandas 的 ewm 计算 EMA12、EMA26、DIF、DEA 和 MACD 柱。布林带使用 rolling(20).mean() 与 rolling(20).std() 计算中轨、上下轨和带宽。ATR 先计算 TR 的三项最大值，再用 14 日均值形成 ATR(14)，并进一步计算 ATR_PCT = ATR / close × 100%。完整代码见 task2_indicator_analysis.py 和 task2_indicator_analysis.ipynb。", body))

    story.append(PageBreak())
    add_heading(story, "五、指标可视化与结果分析", heading)
    add_figure(story, FIG_PRICE, "图1 三只股票收盘价走势与基础价格图", caption, ratio=0.74)
    story.append(p(f"图1显示三只股票的开盘价和收盘价走势。以贵州茅台为例，样本期首日收盘价为 {vals['first_close']:.2f} 元，末日收盘价为 {vals['last_close']:.2f} 元，区间涨跌幅为 {vals['total_ret']:.2f}%。三只股票价格水平差异较大，因此价格图主要适合观察各自趋势，不适合直接比较绝对价格高低。", body))

    add_figure(story, FIG_RSI, "图2 三只股票 RSI(14) 指标图", caption, ratio=0.68)
    story.append(p(f"图2展示 RSI(14) 与 70、30 参考阈值。贵州茅台样本期末最新 RSI 约为 {vals['latest_rsi']:.2f}。当 RSI 接近高位或低位时，说明近期上涨或下跌动能相对集中，但趋势延续会让 RSI 长时间停留在极端区域，因此需要结合价格结构和风险控制解释。", body))

    add_figure(story, FIG_MACD, "图3 三只股票 MACD(12,26,9) 指标图", caption, ratio=0.72)
    story.append(p("图3同时展示三只股票的 DIF、DEA 和 MACD 柱。DIF 与 DEA 的距离变化反映快慢均线差异的扩大或收缩，柱体由正转负或由负转正通常提示动能方向发生变化。由于 MACD 基于均线平滑，优点是能过滤部分短期噪声，局限是滞后，不能替代交易计划。", body))

    add_figure(story, FIG_BB, "图4 三只股票布林带 Bollinger Bands(20,2) 图", caption, ratio=0.74)
    story.append(p("图4显示三只股票的收盘价、中轨 MA20、上轨和下轨。布林带变宽说明近期波动增强，变窄说明波动收敛。价格突破或贴近上下轨时，应更多理解为相对位置和波动状态变化，而非确定性买入或卖出信号。", body))

    add_figure(story, FIG_ATR, "图5 三只股票 ATR_PCT(14) 相对波动率图", caption, ratio=0.68)
    story.append(p(f"图5展示 ATR_PCT，相当于用 ATR(14) 除以收盘价后的相对波动率。贵州茅台样本期末最新值约为 {vals['latest_atr_pct']:.2f}%。ATR 只衡量波动幅度，不判断价格上涨或下跌方向，因此更适合用于风险识别、仓位波动控制或不同股票间波动比较。", body))

    add_heading(story, "六、扩展指标 ATR 分析", heading)
    story.append(p("ATR（Average True Range，平均真实波幅）用于衡量市场波动率。TR 的三项最大值计算方法为：第一，当日最高价 - 当日最低价；第二，abs(当日最高价 - 前一日收盘价)；第三，abs(当日最低价 - 前一日收盘价)。ATR(14) 为 TR 的 14 日移动平均。本报告还计算 ATR_PCT = ATR / close × 100%，用于把不同价格水平的股票放在相对波动率口径下比较。ATR 的重要局限是它不判断涨跌方向，也不说明波动来自利好还是利空。", body))
    add_figure(story, FIG_COMPARE, "图6 三只股票归一化收盘价与 ATR_PCT 横向对比图", caption, ratio=0.66)
    story.append(p("图6把三只股票首日收盘价统一为 100，并同时比较 ATR_PCT。归一化价格用于比较样本期内相对涨跌幅，ATR_PCT 用于比较不同价格水平股票的相对波动。横向比较显示，收益表现和波动水平并不总是同步，单一指标难以完整描述风险收益特征。", body))

    add_heading(story, "七、交互式指标网站说明", heading)
    story.append(p("除正式 PDF 外，本次生成 indicator_dashboard.html 作为补充材料。页面包含主股票贵州茅台以及宁德时代、招商银行的股票选择下拉框，支持全部、近 1 年、近 6 个月、近 3 个月的时间范围切换，支持布林带、RSI、MACD、ATR 的显示开关，并在图表悬停时展示日期、收盘价和对应指标值。HTML 使用本地嵌入数据和原生 Canvas 绘图，不依赖外部 CDN。", body))

    add_heading(story, "八、总结", heading)
    story.append(p("本次任务完成了数据基础诊断、描述性统计、三类基础技术指标和 ATR 扩展指标的计算与可视化。贵州茅台样本数据在本次检查范围内质量较好，可用于课程层面的指标分析。RSI、MACD、布林带和 ATR 分别从相对强弱、趋势动能、价格波动区间和真实波幅角度提供信息，但所有指标都存在滞后、噪声或解释边界。后续若用于策略开发，还应加入复权口径、交易成本、风险约束、回测验证和样本外检验。", body))

    def page_number(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawCentredString(A4[0] / 2, 0.8 * cm, f"第 {doc.page} 页")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.55 * cm,
        title="jane+TASK2",
    )
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)


def make_notebook() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell("# TASK2 数据炼金术：数据诊断与构造交易指标\n\n主股票：贵州茅台（600519.SH）。"),
        nbf.v4.new_markdown_cell("本 Notebook 对应同目录 `task2_indicator_analysis.py`，复用本地 CSV，计算 RSI、MACD、布林带和 ATR，并生成图表、HTML 与 PDF。"),
        nbf.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\n\nBASE_DIR = Path.cwd()\nindicator_data = pd.read_csv(BASE_DIR / 'task2_600519_indicator_data.csv')\nindicator_data.head()"),
        nbf.v4.new_code_cell("indicator_data[['open','high','low','close','pct_chg','vol','amount']].describe().T"),
        nbf.v4.new_markdown_cell("## 指标计算公式\n\n- RSI(14)：涨跌幅 -> 平均上涨/平均下跌 -> RS -> RSI。\n- MACD(12,26,9)：EMA12、EMA26、DIF、DEA、MACD柱。\n- 布林带(20,2)：MA20、上轨、下轨、带宽。\n- ATR(14)：TR 三项最大值的 14 日移动平均；ATR_PCT = ATR / close × 100%。"),
        nbf.v4.new_code_cell("indicator_data[['trade_date','close','rsi_14','dif','dea','macd_bar','bb_middle','bb_upper','bb_lower','atr_14','atr_pct']].tail()"),
        nbf.v4.new_markdown_cell("## 输出图表\n\n![图1](figure1_price_trend.png)\n\n![图2](figure2_rsi.png)\n\n![图3](figure3_macd.png)\n\n![图4](figure4_bollinger_bands.png)\n\n![图5](figure5_atr_pct.png)\n\n![图6](figure6_cross_stock_comparison.png)"),
        nbf.v4.new_markdown_cell("交互式看板文件：`indicator_dashboard.html`。正式提交 PDF：`jane+TASK2.pdf`。"),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NB_PATH)


def make_readme(source_note: str) -> None:
    README_PATH.write_text(
        f"""# TASK2 数据诊断与交易指标构造

本目录为 TASK2 作业输出，主股票为贵州茅台（600519.SH），补充对比股票为宁德时代（300750.SZ）和招商银行（600036.SH）。

## 文件结构

- `jane+TASK2.pdf`：正式提交 PDF 报告。
- `task2_indicator_analysis.py`：完整 Python 生成脚本。
- `task2_indicator_analysis.ipynb`：Notebook 版本与结果浏览。
- `600519_SH_daily_data.csv`：从 TASK1 复用的贵州茅台本地日行情数据。
- `300750_SZ_daily_data.csv`、`600036_SH_daily_data.csv`：用于网页对比的公开日行情数据。
- `task2_600519_indicator_data.csv`：贵州茅台指标计算结果。
- `task2_all_stocks_indicator_data.csv`：三只股票的指标计算结果。
- `figure1_price_trend.png` 至 `figure6_cross_stock_comparison.png`：报告图表。
- `indicator_dashboard.html`：交互式指标看板，可直接用浏览器打开。

## 数据来源

{source_note}。主股票优先复用 TASK1 已保存 CSV；对比股票通过 AkShare 公开 A 股日行情接口获取，并保存为本地 CSV。报告和网页仅用于课程学习，不构成投资建议。

## 运行方式

在本目录执行：

```bash
python3 task2_indicator_analysis.py
```

脚本会读取本地 CSV，计算 RSI(14)、MACD(12,26,9)、布林带(20,2)、ATR(14) 和 ATR_PCT，并重新生成图表、HTML、Notebook、README 和 PDF。

## 网页打开方式

直接双击 `indicator_dashboard.html` 或在浏览器中打开该文件即可。网页使用内嵌数据和原生 Canvas 绘图，不依赖外部 CDN。
""",
        encoding="utf-8",
    )


def privacy_check() -> None:
    forbidden = ["/" + "Users" + "/", "/" + "var" + "/" + "folders" + "/", "TUSHARE_" + "TOKEN", "token" + "="]
    for path in [PDF_PATH, HTML_PATH, README_PATH, NB_PATH, OUTPUT_CSV, ALL_STOCKS_CSV, Path(__file__).resolve()]:
        if not path.exists():
            continue
        if path.suffix.lower() in {".png", ".pdf"}:
            data = path.read_bytes()
            for marker in forbidden[:2]:
                if marker.encode() in data:
                    raise RuntimeError(f"隐私路径检查未通过：{path.name}")
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for marker in forbidden:
                if marker in text:
                    raise RuntimeError(f"隐私内容检查未通过：{path.name} 包含 {marker}")


def main() -> None:
    main_df, all_df, source_note = load_stock_data()
    main_ind = calculate_indicators(main_df)
    all_ind = add_indicators_to_all(all_df)

    main_ind.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    all_ind.to_csv(ALL_STOCKS_CSV, index=False, encoding="utf-8-sig")

    plot_figures(all_ind)
    make_dashboard(all_ind)
    make_notebook()
    make_readme(source_note)
    make_report(all_ind, source_note)
    privacy_check()

    vals = summary_values(main_ind)
    print("TASK2 生成完成")
    print(f"样本范围：{vals['start']} 至 {vals['end']}，共 {vals['rows']} 个交易日")
    print(f"输出 PDF：{PDF_PATH.name}")
    print(f"交互网页：{HTML_PATH.name}")


if __name__ == "__main__":
    main()
