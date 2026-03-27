# CryptoEdge – Regime-Aware Algorithmic Crypto Trading Bot

CryptoEdge is a full-featured algorithmic crypto trading bot built on [Freqtrade](https://www.freqtrade.io/). It detects the current market regime (Bull Trend, Bear Trend, Ranging, Breakout) and activates the appropriate sub-strategy, applying rigorous risk management and signal confluence scoring.

---

## Architecture

```
Regime Detector ──→ EnsembleStrategy (master)
                        ├── TrendFollower   (BULL_TREND)
                        ├── MeanReverter    (RANGING)
                        └── MomentumBreakout (BREAKOUT)
```

### Regimes
| Regime | Condition |
|--------|-----------|
| **BULL_TREND** | close > EMA_50 > EMA_200 AND ADX > 25 |
| **BEAR_TREND** | close < EMA_50 < EMA_200 AND ADX > 25 |
| **RANGING** | ADX < 20 AND BB_width < BB_width_avg |
| **BREAKOUT** | ATR > ATR_avg × 1.5 AND volume > VOL_avg × 2 |

---

## Project Structure

```
├── config/
│   ├── config.json          # Freqtrade configuration
│   ├── pairlist.json        # Trading pair list
│   └── risk_params.json     # Risk management parameters
├── strategies/
│   ├── RegimeDetector.py    # Standalone regime classifier
│   ├── TrendFollower.py     # BULL_TREND strategy
│   ├── MeanReverter.py      # RANGING strategy
│   ├── MomentumBreakout.py  # BREAKOUT strategy
│   ├── EnsembleStrategy.py  # Master controller
│   └── helpers/
│       ├── indicators.py    # Pure numpy/pandas indicators
│       ├── filters.py       # Confluence scoring & filters
│       └── risk_manager.py  # 2% fixed-fractional sizing
├── backtesting/
│   ├── run_backtest.py      # Backtest runner with MUST-PASS checks
│   ├── walk_forward.py      # Walk-forward optimisation
│   ├── monte_carlo.py       # Monte Carlo simulation
│   └── slippage_stress_test.py
├── data/
│   └── validate_data.py     # OHLCV data quality validator
├── monitoring/
│   ├── telegram_bot.py      # Telegram alerts
│   ├── daily_report.py      # Daily P&L reporting
│   └── health_check.py      # Bot & system health checks
├── tests/                   # Pytest test suite
├── docker-compose.yml
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy and edit the config files:
```bash
# Set your exchange API keys via environment variables (see docker-compose.yml)
export EXCHANGE_KEY=your_key
export EXCHANGE_SECRET=your_secret
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Dry-run backtest

```bash
python backtesting/run_backtest.py \
    --strategy EnsembleStrategy \
    --timerange 20220101-20231231 \
    --config config/config.json
```

### 5. Deploy with Docker

```bash
docker-compose up -d
```

---

## Risk Management

- **Position sizing**: 2% fixed-fractional risk per trade
- **Confluence scoring**: Trades only when score ≥ 3 (full size at ≥ 4)
- **Circuit breakers**: Yellow (5%), Orange (10%), Red (15%) drawdown levels
- **Daily loss limit**: Halt trading at 3% daily loss
- **Filters**: Correlation, spread, volume, cooldown

---

## MUST-PASS Backtest Criteria

| Metric | Threshold |
|--------|-----------|
| Win Rate | ≥ 45% |
| Profit Factor | ≥ 1.3 |
| Max Drawdown | ≤ 25% |
| Sharpe Ratio | ≥ 0.5 |
| Total Trades | ≥ 30 |

---

## Slippage Hard Gate

The strategy **must be profitable at 0.10% slippage** or it is blocked from live deployment.

---

## License

MIT