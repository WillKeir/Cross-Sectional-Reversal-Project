# Statistical Arbitrage Project - Impulse Candle Cross-Sectional Reversal Trading Strategies

This repo contains a report showcasing a small collection of cross-sectional reversal trading strategies built around the concept of impulse candles: candles which exhibit significant price movement with high volume. This project was developed as part of the WallStreetQuants quant bootcamp.

Contents:
- `ImpulseReversal.ipynb` – Jupyter notebook containing the full report
- `functionsCode.py` – Python file containing custom functions used for data sourcing, backtesting, and analysis
- `README.md` file

## Key Findings

- The portfolio of strategies achieved an in-sample Sharpe ratio of 4.42 and an out-of-sample Sharpe ratio of 3.40.
- After accounting for transaction costs, in-sample sharpe ratio dropped to 1.00.
- Following strategy adjustments to improve performance after transaction costs, the in-sample Sharpe ratio increased to 1.91, producing an out-of-sample Sharpe ratio of 1.32.
- Regression analysis indicated that the portfolio of strategies consistantly produced statistically significant alpha, with low beta exposure, both in-smaple and out-of-sample.
