const STOCK_META = {
  "300750.SZ": { file: "data/300750_SZ_daily_data.csv", name: "宁德时代", market: "创业板" },
  "600519.SH": { file: "data/600519_SH_daily_data.csv", name: "贵州茅台", market: "上交所" },
  "600036.SH": { file: "data/600036_SH_daily_data.csv", name: "招商银行", market: "上交所" }
};

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cols[i];
    });
    return {
      date: row.trade_date.slice(0, 10),
      open: Number(row.open),
      high: Number(row.high),
      low: Number(row.low),
      close: Number(row.close),
      vol: Number(row.vol || row.volume || 0)
    };
  }).filter((row) => Number.isFinite(row.close));
}

async function loadStock(code) {
  const meta = STOCK_META[code] || STOCK_META["300750.SZ"];
  const res = await fetch(meta.file);
  if (!res.ok) throw new Error(`无法加载数据：${meta.file}`);
  const data = parseCsv(await res.text());
  return { meta, data };
}

function maxPrev(rows, index, field, window) {
  if (index < window) return null;
  let value = -Infinity;
  for (let i = index - window; i < index; i += 1) value = Math.max(value, rows[i][field]);
  return value;
}

function minPrev(rows, index, field, window) {
  if (index < window) return null;
  let value = Infinity;
  for (let i = index - window; i < index; i += 1) value = Math.min(value, rows[i][field]);
  return value;
}

function addIndicators(rows, params) {
  const tr = rows.map((row, i) => {
    if (i === 0) return row.high - row.low;
    const prevClose = rows[i - 1].close;
    return Math.max(row.high - row.low, Math.abs(row.high - prevClose), Math.abs(row.low - prevClose));
  });

  return rows.map((row, i) => {
    const atrStart = i - params.atrWindow + 1;
    const atr = atrStart >= 0 ? tr.slice(atrStart, i + 1).reduce((a, b) => a + b, 0) / params.atrWindow : null;
    const entryHigh = maxPrev(rows, i, "high", params.entryWindow);
    const exitLow = minPrev(rows, i, "low", params.exitWindow);
    return {
      ...row,
      tr: tr[i],
      atr,
      entryHigh,
      exitLow,
      buySignal: entryHigh !== null && row.close > entryHigh,
      exitSignal: exitLow !== null && row.close < exitLow
    };
  });
}

function runTurtleBacktest(rows, params) {
  const data = addIndicators(rows, params);
  let cash = params.initialCash;
  let units = [];
  let nextAddPrice = null;
  const trades = [];
  const equity = [];

  data.forEach((row) => {
    let action = "";
    const stopPrice = units.length ? Math.max(...units.map((u) => u.stopPrice)) : null;
    const exitByChannel = units.length && row.exitSignal;
    const exitByStop = units.length && stopPrice !== null && row.close <= stopPrice;

    if (exitByChannel || exitByStop) {
      const shares = units.reduce((sum, unit) => sum + unit.shares, 0);
      cash += shares * row.close;
      trades.push({
        date: row.date,
        action: "SELL",
        reason: exitByChannel ? "跌破离场通道" : "触发2ATR止损",
        price: row.close,
        shares,
        units: units.length,
        atr: row.atr,
        stopPrice
      });
      units = [];
      nextAddPrice = null;
      action = "SELL";
    } else if (row.atr && row.atr > 0) {
      const shouldEnter = !units.length && row.buySignal;
      const shouldAdd = units.length && units.length < params.maxUnits && nextAddPrice !== null && row.close >= nextAddPrice;
      if (shouldEnter || shouldAdd) {
        const totalShares = units.reduce((sum, unit) => sum + unit.shares, 0);
        const accountValue = cash + totalShares * row.close;
        const unitSize = Math.max(1, Math.floor((accountValue * params.riskPerUnit) / row.atr));
        const affordable = Math.floor(cash / row.close);
        const shares = Math.min(unitSize, affordable);
        if (shares > 0) {
          cash -= shares * row.close;
          const stopForUnit = row.close - params.stopAtr * row.atr;
          units.push({ shares, entryPrice: row.close, atr: row.atr, stopPrice: stopForUnit });
          nextAddPrice = row.close + params.addAtr * row.atr;
          trades.push({
            date: row.date,
            action: shouldEnter ? "BUY" : "ADD",
            reason: shouldEnter ? "通道突破" : "金字塔加仓",
            price: row.close,
            shares,
            units: units.length,
            atr: row.atr,
            stopPrice: stopForUnit
          });
          action = shouldEnter ? "BUY" : "ADD";
        }
      }
    }

    const shares = units.reduce((sum, unit) => sum + unit.shares, 0);
    equity.push({
      date: row.date,
      close: row.close,
      cash,
      shares,
      units: units.length,
      action,
      atr: row.atr,
      entryHigh: row.entryHigh,
      exitLow: row.exitLow,
      totalAsset: cash + shares * row.close
    });
  });

  return { data, equity, trades, metrics: metricsFromEquity(equity, trades, params.initialCash) };
}

function metricsFromEquity(equity, trades, initialCash) {
  const nav = equity.map((row) => row.totalAsset / initialCash);
  const daily = equity.map((row, i) => (i === 0 ? 0 : row.totalAsset / equity[i - 1].totalAsset - 1));
  let peak = -Infinity;
  let maxDrawdown = 0;
  nav.forEach((value) => {
    peak = Math.max(peak, value);
    maxDrawdown = Math.min(maxDrawdown, value / peak - 1);
  });
  const mean = daily.reduce((a, b) => a + b, 0) / daily.length;
  const variance = daily.reduce((sum, value) => sum + (value - mean) ** 2, 0) / daily.length;
  const std = Math.sqrt(variance);
  const sells = trades.filter((trade) => trade.action === "SELL");
  const completed = sells.length;
  let cost = 0;
  let wins = 0;
  trades.forEach((trade) => {
    if (trade.action === "BUY" || trade.action === "ADD") cost += trade.price * trade.shares;
    if (trade.action === "SELL") {
      if (trade.price * trade.shares > cost) wins += 1;
      cost = 0;
    }
  });
  return {
    finalAsset: equity.at(-1).totalAsset,
    cumulativeReturn: nav.at(-1) - 1,
    annualReturn: nav.at(-1) ** (252 / nav.length) - 1,
    maxDrawdown,
    sharpe: std ? (mean / std) * Math.sqrt(252) : 0,
    winRate: completed ? wins / completed : 0,
    tradeCount: trades.length,
    sellCount: sells.length,
    benchmarkReturn: equity.at(-1).close / equity[0].close - 1
  };
}

function fmtPct(value) {
  return `${(value * 100).toFixed(2)}%`;
}

function fmtMoney(value) {
  return `¥${Math.round(value).toLocaleString("zh-CN")}`;
}

function drawLineChart(canvas, series, options = {}) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 54, right: 22, top: 28, bottom: 42 };
  const all = series.flatMap((s) => s.values.filter((v) => Number.isFinite(v)));
  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const x = (i, n) => pad.left + (i / Math.max(n - 1, 1)) * (width - pad.left - pad.right);
  const y = (v) => height - pad.bottom - ((v - min) / span) * (height - pad.top - pad.bottom);

  ctx.strokeStyle = "#263347";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const yy = pad.top + i * ((height - pad.top - pad.bottom) / 4);
    ctx.beginPath();
    ctx.moveTo(pad.left, yy);
    ctx.lineTo(width - pad.right, yy);
    ctx.stroke();
  }

  ctx.fillStyle = "#94a3b8";
  ctx.font = "12px -apple-system, BlinkMacSystemFont, Segoe UI";
  ctx.fillText((max).toFixed(options.decimals ?? 2), 8, pad.top + 4);
  ctx.fillText((min).toFixed(options.decimals ?? 2), 8, height - pad.bottom);

  series.forEach((s) => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = s.width || 2;
    ctx.setLineDash(s.dash || []);
    ctx.beginPath();
    s.values.forEach((value, i) => {
      if (!Number.isFinite(value)) return;
      const xx = x(i, s.values.length);
      const yy = y(value);
      if (i === 0) ctx.moveTo(xx, yy);
      else ctx.lineTo(xx, yy);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  });
}

function drawBarChart(canvas, labels, datasets) {
  const ctx = canvas.getContext("2d");
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  const pad = { left: 54, right: 20, top: 28, bottom: 60 };
  const values = datasets.flatMap((d) => d.values);
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const span = max - min || 1;
  const y = (v) => height - pad.bottom - ((v - min) / span) * (height - pad.top - pad.bottom);
  const zero = y(0);
  ctx.strokeStyle = "#ccd5e3";
  ctx.beginPath();
  ctx.moveTo(pad.left, zero);
  ctx.lineTo(width - pad.right, zero);
  ctx.stroke();

  const groupWidth = (width - pad.left - pad.right) / labels.length;
  const barWidth = Math.min(34, (groupWidth - 18) / datasets.length);
  labels.forEach((label, i) => {
    datasets.forEach((d, j) => {
      const value = d.values[i];
      const x = pad.left + i * groupWidth + (groupWidth - barWidth * datasets.length) / 2 + j * barWidth;
      ctx.fillStyle = d.color;
      ctx.fillRect(x, Math.min(zero, y(value)), barWidth - 4, Math.abs(zero - y(value)));
    });
    ctx.save();
    ctx.translate(pad.left + i * groupWidth + groupWidth / 2, height - 24);
    ctx.rotate(-0.35);
    ctx.fillStyle = "#475467";
    ctx.font = "12px sans-serif";
    ctx.textAlign = "right";
    ctx.fillText(label, 0, 0);
    ctx.restore();
  });
}
