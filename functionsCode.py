# Imports
from binance.client import Client as bnb_client
from datetime import datetime
import pandas as pd 
import numpy as np 

import warnings
warnings.filterwarnings("ignore")
# warnings.filterwarnings("default")


# BNB Client
client = bnb_client()

# Binance data retrieval function
def get_binance_px(symbol, freq, start_ts, end_ts):
    data = client.get_historical_klines(symbol,freq,start_ts,end_ts)
    columns = ['open_time','open','high','low','close','volume','close_time','quote_volume',
    'num_trades','taker_base_volume','taker_quote_volume','ignore']

    data = pd.DataFrame(data,columns = columns)
    
    # Convert from POSIX timestamp (number of millisecond since jan 1, 1970)
    data['open_time'] = data['open_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    data['close_time'] = data['close_time'].map(lambda x: datetime.utcfromtimestamp(x/1000))
    return data 

# get_data function
def get_data(univ, freq, start_ts, end_ts, vars):
    frames = []
    for x in univ:
        data = get_binance_px(x,freq,start_ts, end_ts)
        data = data.set_index('open_time')[vars]
        data.columns = pd.MultiIndex.from_product([[x], data.columns])
        frames.append(data)

    px = pd.concat(frames, axis=1).astype(float)
    px = px.reindex(pd.date_range(px.index[0],px.index[-1],freq=freq))

    px = px.dropna()

    return px

# Function to define an impulse candle on BTC
def impulse(dfSlow, rollLen=14, volmaLen=75, percentile=90, pctShift=0, alt=False, type=1):

    # Copy Data. BTC Data
    data = dfSlow['BTCUSDT'].copy()

    # Rolling Percentile of Returns
    rets = data['close'].pct_change()
    largestUp = rets.rolling(rollLen).apply(lambda x: np.percentile(x, percentile), raw=True).shift(pctShift)

    # Volume analysis
    vol = data['volume']
    volma = data['volume'].rolling(volmaLen).mean()

    # Initialise Signal
    temp = pd.Series(0.0, index=rets.index)
    
    # Impulse Candle Detection
    condition = pd.Series(False, index=rets.index)
    
    # Detect candles that meet certain criteria
    if type == 1:
        condition = (rets > largestUp) & (vol > volma)

    if type == 2:
        condition = ((rets > largestUp) & (vol < volma))

    if type == 3:
        condition = (rets < largestUp) & (vol > volma)

    if type == 4:
        condition = ((rets < largestUp) & (vol < volma))
    
    if type == 5:
        condition = (rets > largestUp)

    if type == 6:
        condition = pd.Series(True, index=rets.index)        
    
    if alt and not type == 6:
        condition = ~condition

    temp = temp.where(~condition, 1.0)

    signal = temp
    signal = signal.replace(0.0, np.nan)

    return signal

# Backtest function
def backtest(dfSlow, dfFast, rollLen=14, volmaLen=75, percentile=90, pctShift=0, alt=False, lags=[-1], split=2, type=1, sigLag=1, tcosts=False):
    
    # Copy data
    assets = list(dfSlow.columns.get_level_values(0).unique())
    dataSlow = dfSlow[assets].copy()
    dataFast = dfFast[assets].copy()

    retsFast = dataFast.xs('close', axis=1, level=1)
    retsFast = retsFast.pct_change()

    retsSlow = dataSlow.xs('close', axis=1, level=1)
    retsSlow = retsSlow.pct_change()

    # Find impulse candles in data based off parameters
    impulseSer = impulse(dataSlow, rollLen, volmaLen, percentile, pctShift, alt, type)

    # Align and reindex impulse confirmation such that appropriate weights can be easily computed
    impulseSer = impulseSer.shift(1)
    impulseSer = impulseSer.reindex(retsFast.index)
    
    # (sigLag = 0) results in weights being computed relative to the last 1hr of price movement before the daily close
    # (sigLag > 0) computes weights using the most recent 1hr candle, (sigLag) hours after the daily close
    impulseSer = impulseSer.shift(-1 + sigLag)

    # Compute weights
    wgts = retsFast.mul(impulseSer, axis=0)
    wgts = wgts.rank(1) 
    port = pd.DataFrame(0.0, index=wgts.index, columns=wgts.columns)

    ## Equal split between each asset, long top performers, short bottom performers
    for row in wgts.index:    
        ranks = wgts.loc[row]
        bottom = list(ranks.nsmallest(split).index)
        top = list(ranks.nlargest(split).index)

        if not np.isnan(ranks[top[0]]):
            port.loc[row, bottom] = -1 / (2 * split)
            port.loc[row, top] = 1 / (2 * split)

    # Compute straetegy returns based on signal, 
    # with the option to account changes in relative weights due to buy and hold criteria
    port_final = np.sign(lags[0]) * port.shift(abs(lags[0]))
    indicator = (port != 0.0).astype(float)
    for i in range(1, len(lags)):
        if tcosts:
            port_final += port_final.shift() * (1+retsFast.shift(1)) * indicator.shift(abs(lags[i]))
        
        else:
            port_final += port.shift(abs(lags[i])) * np.sign(lags[i]) 

    # Make weights sum to 1
    abs_sum = port_final.abs().sum(1)
    abs_sum = abs_sum.replace(0.0, np.nan)
    port_final = port_final.div(abs_sum, axis=0)
    port_final = port_final.fillna(0.0)
    
    # Produce straetegy returns
    ret = port_final * retsFast 
    ret = ret.fillna(0.0, axis=0)
    ret = ret.sum(1)
    
    # Account for transaction costs
    ## Assumption of 7 basis points (limit order entry and exit)
    if tcosts:
        bps = 7
        feeRate = 0.0001 * bps

        for i in range(len(ret)-1):
            if ret.iloc[i] == 0.0 and ret.iloc[i+1] != 0.0:
                ret.iloc[i+1] -= feeRate / 2

            if ret.iloc[i] != 0.0 and ret.iloc[i+1] == 0.0:
                ret.iloc[i] -= feeRate / 2

    return ret

# Simple XS Strategy, for regression analysis
def basic_XS(data):
    # Copy data
    df = data.copy()

    # Seperate close values for each token
    rets = df.xs('close', axis=1, level=1).pct_change()

    # Compute weights: equal weight, long bottom half, short top half
    rank = rets.rank(1)
    rank = rank.sub(rank.mean(1), axis=0)
    rank = np.sign(rank)
    rank = rank.div(rank.abs().sum(1), axis=0)

    # Reverse trade direction for reversal strategy
    port = -rets.mul(rank.shift(1))

    # Compute returns
    strategy = port.sum(1)
    strategy.name = 'Basic XS'

    return strategy

# Compute Performance Metrics of a Strategy
def performance_metrics(rets):
    # Calculations
    ## Sharpe Ratio
    sr = round(rets.mean() / rets.std() * np.sqrt(365 * 24), 3)

    ## Max Drawdown
    cumprod = (1+rets).cumprod()
    hwm = cumprod.cummax()
    dd = (hwm - cumprod)/hwm

    maxDrawdown = round(100*dd.max(), 3)

    ## Win Rate, Average Returns, Average Win, Average Loss, Largest Win, Largest Loss
    start=1.0
    wins, losses = 0, 0
    totalRets, totalWin, totalLoss, maxWin, maxLoss = 0.0, 0.0, 0.0, 0.0, 0.0

    for i in range(1, len(rets)):
        if rets.iloc[i] != 0.0:
            start *= (1+rets.iloc[i])

        if rets.iloc[i] == 0.0 and rets.iloc[i-1] != 0.0:
            if start > 1.0:
                wins+=1
                totalWin+=(start-1)
                if (start-1 > maxWin):
                    maxWin = start-1

            else:
                losses+=1
                totalLoss+=(start-1)
                if (start-1 < maxLoss):
                    maxLoss = start-1

            totalRets += start-1
            start=1.0
    
    winRate = round(100*(wins / (wins + losses)), 3)
    avgRets = round(100*(totalRets/wins), 3)
    avgWin = round(100*(totalWin / wins), 3)
    avgLoss = round(100*(totalLoss / losses), 3)
    largestUp = round(100*maxWin, 3)
    largestDwn = round(100*maxLoss, 3)

    AnnRets = round(100 * (((1+rets).cumprod()[-1]) ** (365*24 / len(rets)) - 1), 3)

    # Combine into a DataFrame
    metrics = pd.DataFrame({
        'Metric': [
            'Sharpe Ratio', 
            'Max Drawdown (%)', 
            'Win Rate (%)',
            'Annualised Returns (%)',
            'Average Returns per Trade (%)',
            'Average Winning Trade (%)',
            'Average Losing Trade (%)',
            'Largest Winning Trade (%)',
            'Largest Losing Trade (%)'
        ],
        'Value': [
            sr,
            maxDrawdown,
            winRate,
            AnnRets,
            avgRets,
            avgWin,
            avgLoss,
            largestUp,
            largestDwn
        ]
    })

    return metrics

