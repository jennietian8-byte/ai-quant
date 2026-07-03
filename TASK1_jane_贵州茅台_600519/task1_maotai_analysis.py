#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TASK1 量化交易初体验：从零搭建数据引擎

本脚本使用 Tushare 获取贵州茅台（600519.SH）过去一年日行情数据，完成清洗、
数据质量检查、绘图，并生成 CSV、Notebook、HTML 看板和 PDF 报告。

Token 安全说明：
- 优先从环境变量 TUSHARE_TOKEN 读取。
- 不在任何输出文件中写入真实 token。
- 如果本机缺少宋体，PDF 与图表将自动使用可用中文字体替代。
"""

from __future__ import annotations

import os
import textwrap
import time
from datetime import datetime
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
import tushare as ts
from matplotlib.ticker import StrMethodFormatter
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
TS_CODE = "600519.SH"
STOCK_NAME = "贵州茅台"
MARKET = "上海证券交易所 A 股主板"

CSV_PATH = BASE_DIR / "600519_SH_daily_data.csv"
PNG_PATH = BASE_DIR / "close_price_curve.png"
HTML_PATH = BASE_DIR / "600519_SH_dashboard.html"
PDF_PATH = BASE_DIR / "jane+TASK1.pdf"
NB_PATH = BASE_DIR / "task1_maotai_analysis.ipynb"

REQUIRED_COLUMNS = [
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "pre_close",
    "change",
    "pct_chg",
    "vol",
    "amount",
]


def choose_chinese_font() -> tuple[str, str | None]:
    """Return a usable Chinese font name and optional file path."""
    candidates = [
        ("Songti SC", "/System/Library/Fonts/Supplemental/Songti.ttc"),
        ("STHeiti", "/System/Library/Fonts/STHeiti Medium.ttc"),
        ("PingFang SC", "/System/Library/Fonts/PingFang.ttc"),
        ("Arial Unicode MS", "/Library/Fonts/Arial Unicode.ttf"),
        ("Noto Sans CJK SC", None),
        ("SimSun", None),
    ]
    available_names = {f.name for f in fm.fontManager.ttflist}
    for name, path in candidates:
        if path and Path(path).exists():
            return name, path
        if name in available_names:
            return name, None
    return "DejaVu Sans", None


CHINESE_FONT_NAME, CHINESE_FONT_PATH = choose_chinese_font()
plt.rcParams["font.sans-serif"] = [CHINESE_FONT_NAME, "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_token() -> str:
    token = os.getenv("TUSHARE_TOKEN", "").strip()
    if not token or token == "YOUR_TUSHARE_TOKEN":
        raise RuntimeError(
            "未检测到有效 TUSHARE_TOKEN。请先设置环境变量，例如："
            "export TUSHARE_TOKEN='YOUR_TUSHARE_TOKEN'"
        )
    return token


def fetch_daily_data() -> tuple[pd.DataFrame, str, str, str]:
    today = pd.Timestamp.today().normalize()
    start = today - pd.DateOffset(years=1)
    start_date = start.strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")

    ts.set_token(get_token())
    pro = ts.pro_api()
    source_note = "Tushare Pro daily 接口"
    try:
        raw = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)
    except Exception as exc:
        raw = None
        source_note = f"Tushare Pro daily 权限不足，使用 AkShare 公开 A 股日行情备用接口补齐字段；原始错误：{exc}"

    if raw is None or raw.empty:
        import akshare as ak

        last_error = None
        ak_raw = None
        for attempt in range(3):
            try:
                ak_raw = ak.stock_zh_a_hist(
                    symbol="600519",
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
                break
            except Exception as exc:
                last_error = exc
                time.sleep(2 + attempt)
        if (ak_raw is None or ak_raw.empty) and CSV_PATH.exists():
            cached = pd.read_csv(CSV_PATH)
            cached["trade_date"] = pd.to_datetime(cached["trade_date"])
            source_note = f"{source_note}；备用接口临时失败，使用本地已生成 CSV 继续生成报告。备用错误：{last_error}"
            return cached.loc[:, REQUIRED_COLUMNS], start_date, end_date, source_note
        if ak_raw is None or ak_raw.empty:
            raise RuntimeError("Tushare Pro 与备用公开行情接口均未返回数据，请检查网络或日期范围。")
        raw = pd.DataFrame(
            {
                "trade_date": pd.to_datetime(ak_raw["日期"]).dt.strftime("%Y%m%d"),
                "open": ak_raw["开盘"],
                "high": ak_raw["最高"],
                "low": ak_raw["最低"],
                "close": ak_raw["收盘"],
                "pre_close": ak_raw["收盘"].shift(1),
                "change": ak_raw["涨跌额"],
                "pct_chg": ak_raw["涨跌幅"],
                "vol": ak_raw["成交量"],
                "amount": ak_raw["成交额"],
            }
        )
        raw["pre_close"] = raw["pre_close"].fillna(raw["close"] - raw["change"])

    df = raw.loc[:, REQUIRED_COLUMNS].copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
    df = df.sort_values("trade_date").reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df, start_date, end_date, source_note


def yes_no(value: bool) -> str:
    return "通过" if value else "不通过"


def make_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    price_cols = ["open", "high", "low", "close"]
    missing = int(df.isna().sum().sum())
    duplicate_dates = int(df["trade_date"].duplicated().sum())
    sorted_ok = bool(df["trade_date"].is_monotonic_increasing)
    non_positive_prices = int((df[price_cols] <= 0).sum().sum())
    negative_volume_amount = int(((df[["vol", "amount"]] < 0).sum()).sum())
    ohlc_bad_mask = ~(
        (df["high"] >= df["open"])
        & (df["high"] >= df["close"])
        & (df["low"] <= df["open"])
        & (df["low"] <= df["close"])
        & (df["high"] >= df["low"])
    )
    ohlc_bad = int(ohlc_bad_mask.sum())
    row_reasonable = len(df) > 150

    rows = [
        [
            "数据行数是否合理",
            "统计清洗后记录数，并与过去一年 A 股约 240 个交易日的经验范围比较",
            f"{len(df)} 行",
            yes_no(row_reasonable),
            "行数大于 150，覆盖过去一年主要交易日，后续仍需结合停牌和节假日核查。",
        ],
        [
            "是否存在缺失值",
            "对全部字段执行 isna().sum()",
            f"缺失值总数 {missing}",
            yes_no(missing == 0),
            "若出现缺失，应追溯字段来源和接口返回，不应直接删除。",
        ],
        [
            "是否存在重复交易日期",
            "检查 trade_date 是否 duplicated",
            f"重复日期 {duplicate_dates} 个",
            yes_no(duplicate_dates == 0),
            "重复日期可能来自重复抓取或合并错误，应先定位来源。",
        ],
        [
            "日期是否按升序排列",
            "检查 trade_date.is_monotonic_increasing",
            "已按交易日期升序排列" if sorted_ok else "未按升序排列",
            yes_no(sorted_ok),
            "升序排列便于绘图、计算收益率和后续回测。",
        ],
        [
            "OHLC 是否存在非正数",
            "检查 open、high、low、close 是否 <= 0",
            f"非正价格 {non_positive_prices} 个",
            yes_no(non_positive_prices == 0),
            "股票价格非正通常不合常识，若出现应核查复权、接口或数据类型。",
        ],
        [
            "成交量和成交额是否存在负数",
            "检查 vol、amount 是否 < 0",
            f"负数 {negative_volume_amount} 个",
            yes_no(negative_volume_amount == 0),
            "负成交量或负成交额通常表示数据错误，应回查原始数据。",
        ],
        [
            "OHLC 逻辑是否合理",
            "检查 high>=open/close、low<=open/close、high>=low",
            f"异常记录 {ohlc_bad} 行",
            yes_no(ohlc_bad == 0),
            "异常是调查线索，不等于垃圾；应对照交易所或数据商记录核查。",
        ],
    ]
    return pd.DataFrame(rows, columns=["检查项目", "检查方法", "检查结果", "是否通过", "说明 / 处理方式"])


def plot_close_curve(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
    ax.plot(df["trade_date"], df["close"], color="#1f6f8b", linewidth=2.0)
    ax.scatter(df["trade_date"].iloc[-1], df["close"].iloc[-1], color="#c0392b", s=26, zorder=3)
    ax.set_title(f"{STOCK_NAME}（{TS_CODE}）过去一年每日收盘价走势", fontsize=15, pad=14)
    ax.set_xlabel("交易日期", fontsize=11)
    ax.set_ylabel("收盘价（元）", fontsize=11)
    ax.text(
        0.02,
        0.92,
        f"{STOCK_NAME} {TS_CODE}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#ffffff", edgecolor="#aaaaaa", alpha=0.88),
    )
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    fig.autofmt_xdate(rotation=25)
    fig.tight_layout()
    fig.savefig(PNG_PATH, bbox_inches="tight")
    plt.close(fig)


def trend_summary(df: pd.DataFrame) -> str:
    first = float(df["close"].iloc[0])
    last = float(df["close"].iloc[-1])
    high = float(df["close"].max())
    low = float(df["close"].min())
    high_date = df.loc[df["close"].idxmax(), "trade_date"].strftime("%Y-%m-%d")
    low_date = df.loc[df["close"].idxmin(), "trade_date"].strftime("%Y-%m-%d")
    total_pct = (last / first - 1) * 100
    daily_ret = df["close"].pct_change()
    max_abs_day = daily_ret.abs().idxmax()
    max_abs_pct = float(daily_ret.loc[max_abs_day] * 100) if pd.notna(max_abs_day) else 0.0
    max_abs_date = df.loc[max_abs_day, "trade_date"].strftime("%Y-%m-%d") if pd.notna(max_abs_day) else "无"
    direction = "上涨" if total_pct > 0 else "下跌" if total_pct < 0 else "基本持平"
    return (
        f"样本期内，贵州茅台收盘价从 {first:.2f} 元变为 {last:.2f} 元，累计变动 {total_pct:.2f}%，"
        f"整体表现为{direction}。期间最高收盘价为 {high:.2f} 元（{high_date}），"
        f"最低收盘价为 {low:.2f} 元（{low_date}）。单日收盘价最大相对波动约为 {max_abs_pct:.2f}%"
        f"（{max_abs_date}）。图形主要用于观察走势和发现可能跳变，不能单独证明数据质量，也不构成投资建议。"
    )


def make_dashboard(df: pd.DataFrame, quality_df: pd.DataFrame) -> None:
    actual_start = df["trade_date"].min().strftime("%Y-%m-%d")
    actual_end = df["trade_date"].max().strftime("%Y-%m-%d")
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    change_pct = (last_close / first_close - 1) * 100
    max_close = float(df["close"].max())
    min_close = float(df["close"].min())
    avg_close = float(df["close"].mean())
    max_date = df.loc[df["close"].idxmax(), "trade_date"].strftime("%Y-%m-%d")
    min_date = df.loc[df["close"].idxmin(), "trade_date"].strftime("%Y-%m-%d")

    quality_rows = "\n".join(
        "          <tr>"
        f"<td>{escape(str(row['检查项目']))}</td>"
        f"<td><span class=\"status\">{escape(str(row['是否通过']))}</span></td>"
        f"<td>{escape(str(row['检查结果']))}；{escape(str(row['说明 / 处理方式']))}</td>"
        "</tr>"
        for _, row in quality_df.iterrows()
    )
    analysis = (
        f"从过去一年收盘价走势看，{STOCK_NAME}在样本期内最高收盘价为 {max_close:.2f} 元"
        f"（{max_date}），最低收盘价为 {min_close:.2f} 元（{min_date}）。"
        f"样本期首日收盘价为 {first_close:.2f} 元，末日收盘价为 {last_close:.2f} 元，"
        f"区间变化约为 {change_pct:.2f}%。这说明在本次样本范围内，价格中枢有所变化，"
        "但单个股票一年的行情不能单独作为交易决策依据，还需要结合市场环境、基本面变化、"
        "交易成本和风险控制规则。"
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{STOCK_NAME}（{TS_CODE}）量化交易数据引擎看板</title>
  <style>
    :root {{
      --ink: #18212f; --muted: #5f6877; --line: #d9dee8; --panel: #fff;
      --page: #f4f7fb; --ok: #0f7a3b; --ok-soft: #e9f8ef; --warn: #8a5a00;
      --warn-soft: #fff6dd;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; color: var(--ink); background: var(--page);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
        "Hiragino Sans GB", "Microsoft YaHei", Arial, sans-serif;
      line-height: 1.6;
    }}
    .wrap {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 36px; }}
    header {{
      padding: 24px; color: #fff;
      background: linear-gradient(135deg, #7f1d1d 0%, #b42318 54%, #d97706 100%);
      border-radius: 8px; box-shadow: 0 12px 28px rgba(80, 44, 37, .18);
    }}
    h1 {{ margin: 0; font-size: clamp(24px, 4vw, 38px); line-height: 1.22; letter-spacing: 0; }}
    header p {{ margin: 10px 0 0; color: rgba(255,255,255,.86); font-size: 15px; }}
    section {{
      margin-top: 18px; padding: 22px; background: var(--panel); border: 1px solid var(--line);
      border-radius: 8px; box-shadow: 0 8px 20px rgba(25, 33, 47, .05);
    }}
    h2 {{ margin: 0 0 14px; font-size: 20px; line-height: 1.3; letter-spacing: 0; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .metric {{ min-height: 92px; padding: 14px; border: 1px solid var(--line); border-radius: 8px; background: #fbfcff; }}
    .label, .info span {{ color: var(--muted); font-size: 13px; }}
    .value {{ margin-top: 8px; font-size: 24px; font-weight: 720; line-height: 1.2; overflow-wrap: anywhere; }}
    .note {{ margin-top: 4px; color: var(--muted); font-size: 12px; }}
    .info {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; overflow: hidden; border: 1px solid var(--line); border-radius: 8px; background: var(--line); }}
    .info div {{ padding: 14px; background: #fff; }}
    .info span, .info strong {{ display: block; }}
    .info strong {{ margin-top: 5px; font-size: 16px; }}
    .chart-frame {{ padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }}
    .chart-frame img {{ display: block; width: 100%; height: auto; border-radius: 6px; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid var(--line); border-radius: 8px; font-size: 14px; }}
    th, td {{ padding: 12px 14px; text-align: left; border-bottom: 1px solid var(--line); }}
    th {{ color: #3a4454; background: #f7f9fc; font-weight: 700; }}
    tr:last-child td {{ border-bottom: 0; }}
    .status {{ display: inline-flex; min-width: 58px; justify-content: center; padding: 3px 9px; border-radius: 999px; color: var(--ok); background: var(--ok-soft); font-weight: 700; font-size: 13px; }}
    .analysis {{ display: grid; grid-template-columns: 1.15fr .85fr; gap: 16px; }}
    .analysis p {{ margin: 0; }}
    .notice {{ padding: 14px; border: 1px solid #f2cf8b; border-radius: 8px; color: var(--warn); background: var(--warn-soft); font-weight: 700; }}
    footer {{ margin-top: 18px; color: var(--muted); font-size: 13px; text-align: center; }}
    @media (max-width: 860px) {{ .grid, .info, .analysis {{ grid-template-columns: 1fr 1fr; }} }}
    @media (max-width: 560px) {{
      .wrap {{ width: min(100% - 20px, 1120px); padding-top: 12px; }}
      header, section {{ padding: 16px; }}
      .grid, .info, .analysis {{ grid-template-columns: 1fr; }}
      table {{ font-size: 13px; }}
      th, td {{ padding: 10px; }}
    }}
  </style>
</head>
<body>
  <main class="wrap">
    <header>
      <h1>{STOCK_NAME}（{TS_CODE}）量化交易数据引擎看板</h1>
      <p>基于过去一年日行情数据生成，用于展示数据获取、清洗、检查和可视化结果。</p>
    </header>
    <section>
      <h2>股票基本信息</h2>
      <div class="info">
        <div><span>股票名称</span><strong>{STOCK_NAME}</strong></div>
        <div><span>股票代码</span><strong>{TS_CODE}</strong></div>
        <div><span>市场</span><strong>{MARKET}</strong></div>
        <div><span>数据时间范围</span><strong>{actual_start} 至 {actual_end}</strong></div>
      </div>
    </section>
    <section>
      <h2>数据概览</h2>
      <div class="grid">
        <div class="metric"><div class="label">总交易日数量</div><div class="value">{len(df)}</div><div class="note">日行情记录</div></div>
        <div class="metric"><div class="label">起始日期</div><div class="value">{actual_start}</div><div class="note">样本第一日</div></div>
        <div class="metric"><div class="label">结束日期</div><div class="value">{actual_end}</div><div class="note">样本最后一日</div></div>
        <div class="metric"><div class="label">最高收盘价</div><div class="value">{max_close:.2f}</div><div class="note">{max_date}</div></div>
        <div class="metric"><div class="label">最低收盘价</div><div class="value">{min_close:.2f}</div><div class="note">{min_date}</div></div>
        <div class="metric"><div class="label">平均收盘价</div><div class="value">{avg_close:.2f}</div><div class="note">按 {len(df)} 个交易日计算</div></div>
        <div class="metric"><div class="label">区间首日收盘价</div><div class="value">{first_close:.2f}</div><div class="note">{actual_start}</div></div>
        <div class="metric"><div class="label">区间末日收盘价</div><div class="value">{last_close:.2f}</div><div class="note">区间变化 {change_pct:.2f}%</div></div>
      </div>
    </section>
    <section>
      <h2>收盘价走势图</h2>
      <div class="chart-frame"><img src="close_price_curve.png" alt="{STOCK_NAME} {TS_CODE} 过去一年每日收盘价走势"></div>
    </section>
    <section>
      <h2>数据质量检查表</h2>
      <table>
        <thead><tr><th>检查项目</th><th>结果</th><th>说明</th></tr></thead>
        <tbody>
{quality_rows}
        </tbody>
      </table>
    </section>
    <section>
      <h2>简短中文分析</h2>
      <div class="analysis">
        <div><p>{escape(analysis)}</p></div>
        <div class="notice">本看板仅用于课程学习和数据分析，不构成投资建议。</div>
      </div>
    </section>
    <footer>数据文件：600519_SH_daily_data.csv ｜ 图表文件：close_price_curve.png ｜ 看板文件：600519_SH_dashboard.html</footer>
  </main>
</body>
</html>
"""
    HTML_PATH.write_text(html, encoding="utf-8")


def register_report_font() -> str:
    # 课程要求宋体；macOS 常见替代为 Songti SC。如缺少宋体，则自动使用可显示中文的字体。
    if CHINESE_FONT_PATH and Path(CHINESE_FONT_PATH).exists():
        try:
            pdfmetrics.registerFont(TTFont("ReportChinese", CHINESE_FONT_PATH))
            return "ReportChinese"
        except Exception:
            pass
    return "Helvetica"


def para(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def add_heading(story: list, text: str, style: ParagraphStyle) -> None:
    story.append(Spacer(1, 0.2 * cm))
    story.append(para(text, style))
    story.append(Spacer(1, 0.12 * cm))


def code_block(code: str, style: ParagraphStyle) -> Table:
    lines = [line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;") for line in code.strip().splitlines()]
    wrapped = "<br/>".join(lines)
    table = Table([[Paragraph(wrapped, style)]], colWidths=[16.4 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f6f7f9")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def make_report(df: pd.DataFrame, quality_df: pd.DataFrame, start_date: str, end_date: str, source_note: str) -> None:
    font_name = register_report_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "ChineseBody",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=9,
        leading=13.5,
        alignment=TA_LEFT,
        firstLineIndent=18,
        spaceAfter=5,
    )
    body_no_indent = ParagraphStyle(
        "ChineseBodyNoIndent",
        parent=body,
        firstLineIndent=0,
        alignment=TA_LEFT,
    )
    title = ParagraphStyle(
        "ChineseTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    subtitle = ParagraphStyle(
        "ChineseSubtitle",
        parent=body_no_indent,
        fontSize=10.5,
        leading=15,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    heading = ParagraphStyle(
        "ChineseHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=17,
        spaceBefore=8,
        spaceAfter=4,
    )
    code_style = ParagraphStyle(
        "CodeStyle",
        parent=styles["Code"],
        fontName=font_name,
        fontSize=7.4,
        leading=10,
        alignment=TA_LEFT,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body_no_indent,
        fontSize=8.5,
        leading=12,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#333333"),
    )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="TASK1 量化交易初体验：从零搭建数据引擎",
    )

    story = [
        para("《TASK1 量化交易初体验：从零搭建数据引擎》", title),
        para(f"股票：{STOCK_NAME}（{TS_CODE}）<br/>市场：{MARKET}", subtitle),
    ]

    add_heading(story, "1. 作业说明与股票选择", heading)
    story.append(para(f"本次作业选择 {STOCK_NAME} 作为研究对象，股票代码为 {TS_CODE}，属于{MARKET}。选择该股票的原因是其交易数据较完整、市场关注度较高，适合作为量化交易数据引擎入门练习的样本。", body))

    add_heading(story, "2. 量化交易的优势与限制", heading)
    story.append(para("量化交易相较于传统手工交易的优势主要包括：第一，能处理更多数据，可以把价格、成交量、财务指标和风险指标纳入统一流程；第二，执行更一致，规则确定后不会因为临场情绪而随意改变；第三，能够进行历史检验，在真实交易前先观察规则在历史样本中的表现；第四，风险规则可以写进系统，例如仓位上限、止损条件和异常数据拦截。", body))
    story.append(para("量化交易也有明显限制：数据可能出错，历史规律可能失效，模型可能过拟合，真实交易与回测也可能不同。例如回测中可以按历史收盘价顺利成交，但真实交易会受到滑点、流动性和交易成本影响。", body))
    story.append(para("课程核心观点是：量化交易不是机器比人聪明，而是把交易想法变成可以重复执行和检查的规则。量化不消灭人的判断，而是把判断从“今天凭感觉买不买”移动到“规则是否合理、证据是否可靠”。", body))

    add_heading(story, "3. 基本概念解释", heading)
    story.append(para("K线：一根日 K 线由开盘价、最高价、最低价、收盘价组成，是一个交易日价格路径的压缩快照；成交量和成交额补充反映交易活跃程度。", body))
    story.append(para("基本面：关注企业经营与估值，例如收入、利润、现金流、行业景气、市盈率、市净率等。", body))
    story.append(para("技术面：关注价格、成交量以及由它们衍生出的形态或指标，例如均线、涨跌幅、波动率等。", body))
    story.append(para("基本面与技术面的关系：二者不是互相否定，而是信息来源不同。基本面提供价值判断，技术面提供时机判断，结合使用可以更全面理解市场。", body))

    add_heading(story, "4. 数据来源与获取方法", heading)
    actual_start = df["trade_date"].min().strftime("%Y-%m-%d")
    actual_end = df["trade_date"].max().strftime("%Y-%m-%d")
    story.append(para(f"本次脚本优先使用 Tushare Python 的 Pro daily 接口获取 A 股日行情数据。请求对象为 {STOCK_NAME}（{TS_CODE}），请求区间为 {start_date} 至 {end_date}；实际返回的可用交易日范围为 {actual_start} 至 {actual_end}。报告和代码示例均不暴露真实 token，运行时从环境变量 TUSHARE_TOKEN 读取。", body))
    if source_note != "Tushare Pro daily 接口":
        story.append(para("本地运行时，当前 token 对 Tushare Pro daily 接口权限不足；为完整生成课程要求字段，脚本在保留 Tushare Pro 可复现代码的同时，使用公开 A 股日行情备用接口补齐本次 CSV 与图表。后续若账号开通 daily 权限，可直接由同一脚本切换回 Tushare Pro 数据。", body))

    add_heading(story, "5. Python 实现过程", heading)
    snippet = """
import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import tushare as ts

token = os.getenv("TUSHARE_TOKEN", "YOUR_TUSHARE_TOKEN")
ts.set_token(token)
pro = ts.pro_api()

df = pro.daily(ts_code="600519.SH", start_date=start_date, end_date=end_date)
df["trade_date"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
df = df.sort_values("trade_date").reset_index(drop=True)
df.to_csv("600519_SH_daily_data.csv", index=False, encoding="utf-8-sig")

quality_df = make_quality_table(df)
plt.plot(df["trade_date"], df["close"])
plt.title("贵州茅台（600519.SH）过去一年每日收盘价走势")
plt.xlabel("交易日期")
plt.ylabel("收盘价（元）")
plt.savefig("close_price_curve.png", dpi=180, bbox_inches="tight")
make_dashboard(df, quality_df)
"""
    story.append(para("关键实现步骤包括导入库、设置 Tushare token、获取日行情数据、日期转换与排序、数据质量检查、绘制收盘价曲线、保存 CSV，并生成可直接在浏览器中打开的 HTML 看板。完整可运行代码见同目录下的 Python 脚本和 Notebook。", body))
    story.append(code_block(snippet, code_style))

    add_heading(story, "6. 数据质量检查结果", heading)
    story.append(para("课程强调的三条数据原则是：可视化是检查工具之一，不是质量证明；异常是调查线索，不等于垃圾；智能体可以帮助找问题，但是否可用必须由人负责判断。基于这些原则，本次对行数、缺失值、重复日期、日期顺序、非正价格、负成交量或成交额以及 OHLC 逻辑进行了系统检查。", body))
    table_data = [[Paragraph(str(x), body_no_indent) for x in quality_df.columns]]
    for _, row in quality_df.iterrows():
        table_data.append([Paragraph(str(row[col]), body_no_indent) for col in quality_df.columns])
    q_table = Table(table_data, colWidths=[2.6 * cm, 4.3 * cm, 2.5 * cm, 1.5 * cm, 5.5 * cm], repeatRows=1)
    q_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(q_table)
    all_pass = bool((quality_df["是否通过"] == "通过").all())
    story.append(Spacer(1, 0.15 * cm))
    story.append(para(f"从检查结果看，本次数据质量检查{'全部通过' if all_pass else '存在未通过项目'}。这说明数据{'基本可用于后续分析' if all_pass else '需要进一步核查后再用于后续分析'}，但可视化不能替代系统检查。", body))

    story.append(PageBreak())
    add_heading(story, "7. 收盘价曲线图与分析", heading)
    story.append(KeepTogether([Image(str(PNG_PATH), width=16.2 * cm, height=8.55 * cm), para(f"图 1  {STOCK_NAME}（{TS_CODE}）过去一年每日收盘价走势", caption)]))
    story.append(para(trend_summary(df), body))

    add_heading(story, "8. HTML 可视化看板", heading)
    story.append(para("除 CSV、Notebook、Python 脚本、PDF 报告和收盘价曲线 PNG 外，本次还生成了 600519_SH_dashboard.html，作为数据引擎的可视化输出。该看板包含股票基本信息、数据概览、收盘价走势图、数据质量检查表和简短中文分析，并明确说明本看板仅用于课程学习和数据分析，不构成投资建议。HTML 文件与图表文件放在同一目录下，可直接在浏览器中打开。", body))

    add_heading(story, "9. 总结与反思", heading)
    story.append(para(f"本次数据在行数、缺失值、重复日期、日期顺序、价格正数、成交量成交额非负以及 OHLC 逻辑方面完成检查，结论为{'基本可靠' if all_pass else '仍需核查'}。可视化帮助我们直观看到收盘价的整体趋势、阶段性波动和可能的异常跳变，但它只是发现问题的入口，不是质量证明。", body))
    story.append(para("数据检查很重要，因为量化交易的后续信号、回测和风险控制都建立在数据之上。如果基础数据存在缺失、重复或逻辑错误，模型输出可能看似精确，实际却不可用。", body))
    story.append(para("本次导出的 CSV 数据文件可以作为后续量化交易任务的基础，例如计算收益率、均线、波动率或构建简单回测。但在进入真实交易或更复杂策略前，仍应继续核查复权处理、交易成本、停牌信息和数据供应商差异。", body))
    story.append(para("智能体可以帮助完成代码编写、数据检查、图表生成、报告排版和异常线索整理；但数据是否可用、异常是否需要保留、规则是否合理、结论是否能用于交易决策，仍需要由人负责判断。", body))

    def add_page_number(canvas, doc_obj):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.drawCentredString(A4[0] / 2, 0.85 * cm, f"第 {doc_obj.page} 页")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


def make_notebook(start_date: str, end_date: str) -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(
            "# TASK1 量化交易初体验：从零搭建数据引擎\n\n"
            f"股票：{STOCK_NAME}（{TS_CODE}）  \n市场：{MARKET}\n\n"
            "本 Notebook 使用 Tushare 获取过去一年日行情数据，完成清洗、数据质量检查和收盘价曲线绘制。"
        ),
        nbf.v4.new_markdown_cell(
            "## 量化交易的优势与限制\n\n"
            "优势：能处理更多数据；执行更一致；能够进行历史检验；风险规则可以写进系统。\n\n"
            "限制：数据可能出错；历史规律可能失效；模型可能过拟合；真实交易与回测可能不同。\n\n"
            "课程核心观点：量化交易不是机器比人聪明，而是把交易想法变成可以重复执行和检查的规则。"
            "量化不消灭人的判断，而是把判断从“今天凭感觉买不买”移动到“规则是否合理、证据是否可靠”。"
        ),
        nbf.v4.new_markdown_cell(
            "## 基本概念\n\n"
            "- K线：一根日 K 线由开盘价、最高价、最低价、收盘价组成，是一个交易日价格路径的压缩快照；成交量和成交额补充反映交易活跃程度。\n"
            "- 基本面：关注企业经营与估值，例如收入、利润、现金流、行业景气、市盈率、市净率等。\n"
            "- 技术面：关注价格、成交量以及由它们衍生出的形态或指标，例如均线、涨跌幅、波动率等。\n"
            "- 基本面与技术面的关系：二者不是互相否定，而是信息来源不同。基本面提供价值判断，技术面提供时机判断，结合使用可以更全面理解市场。"
        ),
        nbf.v4.new_code_cell(
            """import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tushare as ts

TS_CODE = "600519.SH"
STOCK_NAME = "贵州茅台"
BASE_DIR = Path.cwd()
CSV_PATH = BASE_DIR / "600519_SH_daily_data.csv"
PNG_PATH = BASE_DIR / "close_price_curve.png"
HTML_PATH = BASE_DIR / "600519_SH_dashboard.html"

token = os.getenv("TUSHARE_TOKEN", "YOUR_TUSHARE_TOKEN")
if token == "YOUR_TUSHARE_TOKEN":
    raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN，再运行 Notebook。")
ts.set_token(token)
pro = ts.pro_api()"""
        ),
        nbf.v4.new_code_cell(
            f"""start_date = "{start_date}"
end_date = "{end_date}"

try:
    df = pro.daily(ts_code=TS_CODE, start_date=start_date, end_date=end_date)
except Exception as exc:
    print(f"Tushare Pro daily 权限不足，使用公开 A 股日行情备用接口。原始错误：{{exc}}")
    import akshare as ak
    last_error = None
    ak_raw = None
    for attempt in range(3):
        try:
            ak_raw = ak.stock_zh_a_hist(symbol="600519", period="daily", start_date=start_date, end_date=end_date, adjust="")
            break
        except Exception as fallback_exc:
            import time
            last_error = fallback_exc
            time.sleep(2 + attempt)
    if ak_raw is not None and not ak_raw.empty:
        df = pd.DataFrame({{
            "trade_date": pd.to_datetime(ak_raw["日期"]).dt.strftime("%Y%m%d"),
            "open": ak_raw["开盘"],
            "high": ak_raw["最高"],
            "low": ak_raw["最低"],
            "close": ak_raw["收盘"],
            "pre_close": ak_raw["收盘"].shift(1),
            "change": ak_raw["涨跌额"],
            "pct_chg": ak_raw["涨跌幅"],
            "vol": ak_raw["成交量"],
            "amount": ak_raw["成交额"],
        }})
        df["pre_close"] = df["pre_close"].fillna(df["close"] - df["change"])
    elif CSV_PATH.exists():
        print(f"备用接口临时失败，读取本地 CSV 继续。备用错误：{{last_error}}")
        df = pd.read_csv(CSV_PATH)
    else:
        raise RuntimeError(f"未获取到日行情数据，备用错误：{{last_error}}")

if df is None or df.empty:
    raise RuntimeError("未获取到日行情数据。")
df = df[["trade_date", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]].copy()
df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str).str.replace("-", "", regex=False), format="%Y%m%d")
df = df.sort_values("trade_date").reset_index(drop=True)
df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
df.head(), df.tail(), df.shape"""
        ),
        nbf.v4.new_markdown_cell(
            "## 数据质量检查原则\n\n"
            "1. 可视化是检查工具之一，不是质量证明。\n"
            "2. 异常是调查线索，不等于垃圾。\n"
            "3. 智能体可以帮助找问题，但是否可用必须由人负责判断。"
        ),
        nbf.v4.new_code_cell(
            """def make_quality_table(df):
    price_cols = ["open", "high", "low", "close"]
    missing = int(df.isna().sum().sum())
    duplicate_dates = int(df["trade_date"].duplicated().sum())
    sorted_ok = bool(df["trade_date"].is_monotonic_increasing)
    non_positive_prices = int((df[price_cols] <= 0).sum().sum())
    negative_volume_amount = int(((df[["vol", "amount"]] < 0).sum()).sum())
    ohlc_bad = int((~(
        (df["high"] >= df["open"]) &
        (df["high"] >= df["close"]) &
        (df["low"] <= df["open"]) &
        (df["low"] <= df["close"]) &
        (df["high"] >= df["low"])
    )).sum())
    yn = lambda x: "通过" if x else "不通过"
    return pd.DataFrame([
        ["数据行数是否合理", "统计记录数并与过去一年交易日经验范围比较", f"{len(df)} 行", yn(len(df) > 150), "行数大于 150，基本合理。"],
        ["是否存在缺失值", "isna().sum()", f"缺失值总数 {missing}", yn(missing == 0), "若出现缺失，应回查来源。"],
        ["是否存在重复交易日期", "trade_date.duplicated()", f"重复日期 {duplicate_dates} 个", yn(duplicate_dates == 0), "重复日期需定位来源。"],
        ["日期是否按升序排列", "is_monotonic_increasing", "已升序" if sorted_ok else "未升序", yn(sorted_ok), "升序便于后续计算。"],
        ["OHLC 是否存在非正数", "open/high/low/close <= 0", f"非正价格 {non_positive_prices} 个", yn(non_positive_prices == 0), "非正价格应核查。"],
        ["成交量和成交额是否存在负数", "vol/amount < 0", f"负数 {negative_volume_amount} 个", yn(negative_volume_amount == 0), "负值应核查。"],
        ["OHLC 逻辑是否合理", "high、low 与 open/close 的逻辑关系", f"异常记录 {ohlc_bad} 行", yn(ohlc_bad == 0), "异常是线索，不应直接删除。"],
    ], columns=["检查项目", "检查方法", "检查结果", "是否通过", "说明 / 处理方式"])

quality_df = make_quality_table(df)
quality_df"""
        ),
        nbf.v4.new_code_cell(
            """plt.rcParams["font.sans-serif"] = ["Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(11, 5.8), dpi=160)
ax.plot(df["trade_date"], df["close"], color="#1f6f8b", linewidth=2)
ax.set_title("贵州茅台（600519.SH）过去一年每日收盘价走势")
ax.set_xlabel("交易日期")
ax.set_ylabel("收盘价（元）")
ax.text(0.02, 0.92, "贵州茅台 600519.SH", transform=ax.transAxes)
ax.grid(True, linestyle="--", alpha=0.35)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.autofmt_xdate(rotation=25)
fig.tight_layout()
fig.savefig(PNG_PATH, bbox_inches="tight")
plt.close(fig)
PNG_PATH"""
        ),
        nbf.v4.new_code_cell(
            """from html import escape

def make_dashboard(df, quality_df):
    actual_start = df["trade_date"].min().strftime("%Y-%m-%d")
    actual_end = df["trade_date"].max().strftime("%Y-%m-%d")
    first_close = float(df["close"].iloc[0])
    last_close = float(df["close"].iloc[-1])
    change_pct = (last_close / first_close - 1) * 100
    max_close = float(df["close"].max())
    min_close = float(df["close"].min())
    avg_close = float(df["close"].mean())
    max_date = df.loc[df["close"].idxmax(), "trade_date"].strftime("%Y-%m-%d")
    min_date = df.loc[df["close"].idxmin(), "trade_date"].strftime("%Y-%m-%d")
    rows = "\\n".join(
        f"<tr><td>{escape(str(row['检查项目']))}</td><td>{escape(str(row['是否通过']))}</td>"
        f"<td>{escape(str(row['检查结果']))}；{escape(str(row['说明 / 处理方式']))}</td></tr>"
        for _, row in quality_df.iterrows()
    )
    html = f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>贵州茅台（600519.SH）量化交易数据引擎看板</title>
  <style>
    body {{ margin: 0; background: #f4f7fb; color: #18212f; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif; line-height: 1.6; }}
    main {{ width: min(1120px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 36px; }}
    header {{ padding: 24px; color: #fff; background: linear-gradient(135deg, #7f1d1d 0%, #b42318 54%, #d97706 100%); border-radius: 8px; }}
    h1 {{ margin: 0; font-size: clamp(24px, 4vw, 38px); letter-spacing: 0; }}
    section {{ margin-top: 18px; padding: 22px; background: #fff; border: 1px solid #d9dee8; border-radius: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }}
    .card {{ padding: 14px; border: 1px solid #d9dee8; border-radius: 8px; background: #fbfcff; }}
    .label {{ color: #5f6877; font-size: 13px; }}
    .value {{ margin-top: 6px; font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }}
    img {{ display: block; width: 100%; height: auto; border: 1px solid #d9dee8; border-radius: 8px; }}
    table {{ width: 100%; border-collapse: collapse; border: 1px solid #d9dee8; }}
    th, td {{ padding: 12px; border-bottom: 1px solid #d9dee8; text-align: left; }}
    th {{ background: #f7f9fc; }}
    .notice {{ padding: 14px; border: 1px solid #f2cf8b; border-radius: 8px; color: #8a5a00; background: #fff6dd; font-weight: 700; }}
    @media (max-width: 760px) {{ .grid {{ grid-template-columns: 1fr; }} main {{ width: calc(100% - 20px); }} }}
  </style>
</head>
<body>
  <main>
    <header><h1>贵州茅台（600519.SH）量化交易数据引擎看板</h1><p>基于过去一年日行情数据生成。</p></header>
    <section><h2>股票基本信息</h2><p>股票名称：贵州茅台；代码：600519.SH；市场：上海证券交易所 A 股主板；数据时间范围：{actual_start} 至 {actual_end}</p></section>
    <section><h2>数据概览</h2><div class="grid">
      <div class="card"><div class="label">总交易日数量</div><div class="value">{len(df)}</div></div>
      <div class="card"><div class="label">最高收盘价</div><div class="value">{max_close:.2f}</div><div>{max_date}</div></div>
      <div class="card"><div class="label">最低收盘价</div><div class="value">{min_close:.2f}</div><div>{min_date}</div></div>
      <div class="card"><div class="label">平均收盘价</div><div class="value">{avg_close:.2f}</div></div>
      <div class="card"><div class="label">起始日期</div><div class="value">{actual_start}</div></div>
      <div class="card"><div class="label">结束日期</div><div class="value">{actual_end}</div></div>
      <div class="card"><div class="label">区间首日收盘价</div><div class="value">{first_close:.2f}</div></div>
      <div class="card"><div class="label">区间末日收盘价</div><div class="value">{last_close:.2f}</div><div>{change_pct:.2f}%</div></div>
    </div></section>
    <section><h2>收盘价走势图</h2><img src="close_price_curve.png" alt="贵州茅台 600519.SH 过去一年每日收盘价走势"></section>
    <section><h2>数据质量检查表</h2><table><thead><tr><th>检查项目</th><th>结果</th><th>说明</th></tr></thead><tbody>{rows}</tbody></table></section>
    <section><h2>简短中文分析</h2><p>样本期首日收盘价为 {first_close:.2f} 元，末日收盘价为 {last_close:.2f} 元，区间变化约为 {change_pct:.2f}%。走势图可用于观察趋势和可能异常，但不能单独作为交易决策依据。</p><div class="notice">本看板仅用于课程学习和数据分析，不构成投资建议。</div></section>
  </main>
</body>
</html>'''
    HTML_PATH.write_text(html, encoding="utf-8")
    return HTML_PATH

make_dashboard(df, quality_df)"""
        ),
        nbf.v4.new_markdown_cell(
            f"![贵州茅台收盘价曲线](close_price_curve.png)\n\n"
            "上图展示贵州茅台过去一年每日收盘价走势，可用于观察整体趋势、阶段性波动和可能跳变。"
            "图形只是数据检查工具之一，不构成投资建议。\n\n"
            "`600519_SH_dashboard.html` 已作为可直接在浏览器中打开的 HTML 看板生成。"
        ),
        nbf.v4.new_markdown_cell(
            "## 总结与反思\n\n"
            "本次 CSV 可以作为后续计算收益率、均线、波动率和简单回测的基础。"
            "但在真实策略研究中，还需要继续核查复权、停牌、交易成本和不同数据源差异。"
            "智能体可以帮助完成代码、检查和排版，但数据是否可用、异常如何处理、规则是否合理，仍应由人负责判断。"
        ),
    ]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    nbf.write(nb, NB_PATH)


def token_leak_check(token: str) -> None:
    for path in [CSV_PATH, PNG_PATH, HTML_PATH, PDF_PATH, NB_PATH, Path(__file__).resolve()]:
        data = path.read_bytes()
        if token.encode() in data:
            raise RuntimeError(f"检测到真实 token 出现在输出文件中：{path}")


def main() -> None:
    df, start_date, end_date, source_note = fetch_daily_data()
    quality_df = make_quality_table(df)
    plot_close_curve(df)
    make_dashboard(df, quality_df)
    make_notebook(start_date, end_date)
    make_report(df, quality_df, start_date, end_date, source_note)
    token_leak_check(get_token())

    actual_start = df["trade_date"].min().strftime("%Y-%m-%d")
    actual_end = df["trade_date"].max().strftime("%Y-%m-%d")
    all_pass = bool((quality_df["是否通过"] == "通过").all())
    print("TASK1 生成完成")
    print(f"CSV: {CSV_PATH}")
    print(f"PNG: {PNG_PATH}")
    print(f"HTML: {HTML_PATH}")
    print(f"PDF: {PDF_PATH}")
    print(f"Notebook: {NB_PATH}")
    print(f"实际数据起止日期: {actual_start} 至 {actual_end}")
    print(f"数据行数: {len(df)}")
    print(f"数据质量检查: {'全部通过' if all_pass else '存在未通过项目'}")


if __name__ == "__main__":
    main()
