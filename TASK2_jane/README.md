# TASK2 数据诊断与交易指标构造

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

TASK1 本地 CSV；对比股票使用 AkShare 公开 A 股日行情接口。主股票优先复用 TASK1 已保存 CSV；对比股票通过 AkShare 公开 A 股日行情接口获取，并保存为本地 CSV。报告和网页仅用于课程学习，不构成投资建议。

## 运行方式

在本目录执行：

```bash
python3 task2_indicator_analysis.py
```

脚本会读取本地 CSV，计算 RSI(14)、MACD(12,26,9)、布林带(20,2)、ATR(14) 和 ATR_PCT，并重新生成图表、HTML、Notebook、README 和 PDF。

## 网页打开方式

直接双击 `indicator_dashboard.html` 或在浏览器中打开该文件即可。网页使用内嵌数据和原生 Canvas 绘图，不依赖外部 CDN。
