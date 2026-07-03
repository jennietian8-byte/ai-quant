# TASK1 量化交易初体验：从零搭建数据引擎

本项目以贵州茅台（600519.SH）为样本，完成过去一年日行情数据获取、清洗、数据质量检查、可视化和报告生成。

## 文件结构

```text
TASK1_jane_贵州茅台_600519/
├── README.md
├── task1_maotai_analysis.ipynb
├── task1_maotai_analysis.py
├── 600519_SH_daily_data.csv
├── close_price_curve.png
├── 600519_SH_dashboard.html
└── jane+TASK1.pdf
```

## 主要内容

- `600519_SH_daily_data.csv`：清洗后的贵州茅台过去一年日行情数据。
- `close_price_curve.png`：过去一年每日收盘价走势图。
- `600519_SH_dashboard.html`：可直接在浏览器中打开的 HTML 数据看板。
- `task1_maotai_analysis.py`：可独立运行的 Python 脚本。
- `task1_maotai_analysis.ipynb`：Jupyter Notebook 分析过程。
- `jane+TASK1.pdf`：课程作业 PDF 报告。

## 运行方式

如需重新获取数据，请先在本地设置 Tushare token：

```bash
export TUSHARE_TOKEN="YOUR_TUSHARE_TOKEN"
python3 task1_maotai_analysis.py
```

脚本优先使用 Tushare Pro daily 接口；若接口权限不足，会使用公开 A 股日行情备用接口或本地 CSV 兜底生成图表、看板和报告。

## 说明

本项目仅用于课程学习和数据分析，不构成投资建议。仓库中不包含真实 Tushare token、`.env` 文件、本机绝对路径或个人本机用户名。
