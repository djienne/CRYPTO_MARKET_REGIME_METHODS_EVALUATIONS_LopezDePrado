# Crypto Regime-Change Detection

This repository contains a reproducible empirical report on crypto market
structure-break diagnostics inspired by Chapter 17 of Marcos Lopez de Prado's
*Advances in Financial Machine Learning*.

The analysis uses Binance daily OHLCV data for BTC, ETH, ETC, SOL, and HYPE.
Spot markets are used where Binance spot history is available; HYPE falls back
to Binance USD-M futures because Binance spot `HYPEUSDT` was unavailable when
the data was downloaded.

## Outputs

- `regime_detection_crypto_report.pdf` is the rendered report.
- `regime_detection_crypto_report.tex` is the LaTeX source.
- `scripts/generate_daily_regime_chapter.py` downloads data, computes
  diagnostics, runs simple long/flat backtests, and regenerates figures/tables.
- `data/binance_daily/` contains downloaded daily data plus diagnostic and
  strategy CSVs.
- `regime_chapter_figs/` contains all generated figures.

## BTC Backtest Example

The trading section converts structure-break estimates into simple long/flat
signals. Signals are computed using information available through close `t`,
positions are shifted by one day, and the strategy trades the next daily
close-to-close return.

<p>
  <img src="regime_chapter_figs/fig_strategy_pnl_BTC.png" alt="BTC long/flat strategy PnL curves" width="640">
</p>

The strategy ranking is in-sample and should be treated as indicative only.
Choosing the best-performing signal after seeing all PnLs introduces
model-selection/data-snooping risk even though the individual signal returns are
shifted to avoid same-close look-ahead.

## Reproduce

```bash
python3 scripts/generate_daily_regime_chapter.py
pdflatex -interaction=nonstopmode regime_detection_crypto_report.tex
pdflatex -interaction=nonstopmode regime_detection_crypto_report.tex
```

The generator writes strict JSON metadata to `regime_chapter_results.json`.
