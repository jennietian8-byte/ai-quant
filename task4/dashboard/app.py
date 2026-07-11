#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Streamlit playground for TASK4 turtle strategy."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from task4_turtle_strategy import INITIAL_CASH, load_all_data, run_turtle_backtest


st.set_page_config(page_title="海龟法则 Playground", layout="wide")
st.title("海龟法则 Playground")

all_data = load_all_data()
code_to_name = {code: df["stock_name"].iloc[0] for code, df in all_data.items()}

with st.sidebar:
    st.header("参数")
    code = st.selectbox("标的", list(all_data.keys()), format_func=lambda value: f"{code_to_name[value]} ({value})")
    entry_window = st.slider("入场突破周期 N", 10, 60, 20, step=5)
    exit_window = st.slider("退出通道周期 M", 5, 30, 10, step=5)
    atr_window = st.slider("ATR 周期", 10, 30, 20, step=2)
    risk_per_unit = st.slider("单单位风险", 0.005, 0.03, 0.01, step=0.005, format="%.3f")
    max_units = st.slider("最大单位数", 1, 4, 4)

df = all_data[code]
work, equity, metrics, trades = run_turtle_backtest(
    df,
    entry_window=entry_window,
    exit_window=exit_window,
    atr_window=atr_window,
    risk_per_unit=risk_per_unit,
    max_units=max_units,
)

kpis = st.columns(6)
kpis[0].metric("年化收益", f"{metrics['annual_return']:.2%}")
kpis[1].metric("夏普比率", f"{metrics['sharpe_ratio']:.2f}")
kpis[2].metric("最大回撤", f"{metrics['max_drawdown']:.2%}")
kpis[3].metric("胜率", f"{metrics['win_rate']:.2%}")
kpis[4].metric("交易笔数", f"{int(metrics['trade_count'])}")
kpis[5].metric("止损/卖出", f"{int(metrics['sell_count'])}")

tab_equity, tab_signal, tab_trades = st.tabs(["净值曲线", "交易信号", "交易记录"])

with tab_equity:
    nav = equity["total_asset"] / INITIAL_CASH
    benchmark = df["close"] / df["close"].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["trade_date"], y=nav, name="海龟策略"))
    fig.add_trace(go.Scatter(x=df["trade_date"], y=benchmark, name="买入持有", line=dict(dash="dot")))
    fig.update_layout(template="plotly_dark", height=520, yaxis_title="净值", margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_signal:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=work["trade_date"], y=work["close"], name="收盘价", line=dict(color="#d8dee9")))
    fig.add_trace(go.Scatter(x=work["trade_date"], y=work["donchian_high"], name="入场上轨", line=dict(color="#ff6b6b")))
    fig.add_trace(go.Scatter(x=work["trade_date"], y=work["donchian_low"], name="退出下轨", line=dict(color="#51cf66")))
    if not trades.empty:
        buy_dates = trades.loc[trades["action"].isin(["BUY", "ADD"]), "trade_date"]
        sell_dates = trades.loc[trades["action"] == "SELL", "trade_date"]
        buys = work[work["trade_date"].isin(pd.to_datetime(buy_dates))]
        sells = work[work["trade_date"].isin(pd.to_datetime(sell_dates))]
        fig.add_trace(go.Scatter(x=buys["trade_date"], y=buys["close"], name="入场/加仓", mode="markers", marker=dict(symbol="triangle-up", size=11, color="#ff3366")))
        fig.add_trace(go.Scatter(x=sells["trade_date"], y=sells["close"], name="离场", mode="markers", marker=dict(symbol="triangle-down", size=11, color="#20c997")))
    fig.update_layout(template="plotly_dark", height=560, yaxis_title="价格", margin=dict(l=20, r=20, t=35, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab_trades:
    st.dataframe(trades, use_container_width=True)
