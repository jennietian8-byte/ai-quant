# TASK4 复刻传奇：海龟交易法则实战演练

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

预览静态门户站：

```bash
python3 -m http.server 8000
```

然后打开 `http://localhost:8000/site/index.html`。

启动互动 Playground：

```bash
pip install -r requirements-dashboard.txt
streamlit run dashboard/app.py
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
- `outputs/multi_stock_metrics.csv`：多标的回测指标
- `outputs/parameter_sensitivity.csv`：参数热力图扫描结果
- `outputs/figures/`：可视化图表
- `site/`：纯静态 AI Quant Lab 门户站，可用于 GitHub Pages
- `site/github-actions/task4-daily-data-deploy.yml.example`：每日构建与 Pages 部署工作流示例
- `dashboard/app.py`：互动式海龟策略 Playground
- `requirements-dashboard.txt`：Playground 依赖
- `task4_turtle_strategy.ipynb`：Notebook 入口

## 主策略结果摘要

- 最终资产：94,515.54
- 累计回报：-5.48%
- 最大回撤：-22.73%
- 夏普比率：-0.26

## 安全说明

本目录不写入真实 Tushare token、MCP server URL、`.env` 内容、浏览器账号信息或本机绝对路径。
