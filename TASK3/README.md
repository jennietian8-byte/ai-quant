# TASK3 策略首秀：用均线交叉反应市场趋势变化

本目录完成双均线交叉策略作业，复用仓库中已经保存的 A 股日行情 CSV，生成交易信号、逐日回测、绩效指标、图表、Notebook 和 PDF 报告。

## 文件结构

```text
TASK3/
├── README.md
├── task3_ma_cross_strategy.py
├── task3_ma_cross_strategy.ipynb
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

脚本默认复用本地 CSV，不需要写入真实 Tushare token。若未来需要重新获取数据，请只通过本地环境变量传入：

```bash
export TUSHARE_TOKEN="YOUR_TUSHARE_TOKEN"
```

## 主实验结果

- 标的：宁德时代（300750.SZ）
- 参数：MA5/MA15
- 初始资金：100000 元
- 手续费：0.03%
- 滑点：0.02%
- 累计回报：0.26%
- 年化收益：0.27%
- 最大回撤：-29.34%
- 夏普比率：0.05
- 超额收益：-44.45%

## 说明

本项目仅用于课程学习和量化策略入门练习，不构成投资建议。仓库中不包含真实 Tushare token、完整 MCP URL、`.env` 文件、本机绝对路径或个人本机用户名。
