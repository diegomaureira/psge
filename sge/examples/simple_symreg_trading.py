import numpy as np
import sge
from backtesting import Backtest
from backtesting.test import GOOG  # or use your own `bk_data`
from backtesting.lib import crossover, Strategy
import warnings
import pandas as pd
import ta
from binance.client import Client
import dotenv
import os

# Load environment variables from .env file
dotenv.load_dotenv()

version = 'demo'

if version == 'demo':
    # Initialize API (replace with your keys)
    api_key = os.getenv('BINANCE_API_TEST_KEY')
    api_secret = os.getenv('BINANCE_API_TEST_SECRET')
    url = 'https://testnet.binance.vision'
    testnet = True
elif version == 'real':
    # Initialize API (replace with your keys)
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    url = 'https://api.binance.com'
    testnet = False

# API key/secret are required for user data endpoints
client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)

import pandas as pd

# Fetch historical data
# get historical kline data from any date range

# fetch 1 minute klines for the last day up until now
klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

# fetch 30 minute klines for the last month of 2017
#klines = client.get_historical_klines("ETHBTC", Client.KLINE_INTERVAL_30MINUTE, "1 Dec, 2017", "1 Jan, 2018")

# fetch weekly klines since it listed
#klines = client.get_historical_klines("NEOBTC", Client.KLINE_INTERVAL_1WEEK, "1 Jan, 2017")

data = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
data = data.astype(float)
# timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
data['close_time'] = pd.to_datetime(data['close_time'], unit='ms')

# Set data to backtesting format
#range = [-1440, -1]
bk_data = data[['open', 'high', 'low', 'close', 'volume']]#[range[0]:range[1]]
bk_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
bk_data.index = pd.to_datetime(data['timestamp'], unit='ms')

def SMA(values, n):
    """
    Return simple moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).rolling(n).mean()

def EMA(values, n):
    """
    Return exponential moving average of `values`, at
    each step taking into account `n` previous values.
    """
    return pd.Series(values).ewm(span=n).mean()

def RSI(values, n):
    """
    Return relative strength index of `values`, at
    each step taking into account `n` previous values.
    """
    values = pd.Series(values)
    low_min = values.rolling(window=n).min()
    high_max = values.rolling(window=n).max()
    rsi = 100 * (values - low_min) / (high_max - low_min)
    return rsi

def ATR(high, low, close, n):
    """
    Return average true range of `high`, `low` and `close`,
    at each step taking into account `n` previous values.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    return ta.volatility.average_true_range(high, low, close, window=n)

class MyStrategy(Strategy):
    def init(self):

        self.ema9 = self.I(EMA, self.data.Close, 9)
        self.ema21 = self.I(EMA, self.data.Close, 21)
        self.ema200 = self.I(EMA, self.data.Close, 200)
        self.rsi = self.I(RSI, self.data.Close, 14)
        self.atr = self.I(ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.tp = 0
        self.sl = 0
        self.entry_price = 0
        
    def next(self):
        close = self.data.Close[-1]
        atr = self.atr[-1]
        rsi = self.rsi[-1]
        prev_rsi = self.rsi[-2]
        ema9 = self.ema9[-1]
        ema21 = self.ema21[-1]
        ema200 = self.ema200[-1]
        
        bullish_trend = close > ema200
        ema_crossover = ema9 > ema21
        volatility_ok = atr > 0.0005
        
        if not self.position:
            if bullish_trend and ema_crossover and (30 < rsi < 70) and (prev_rsi < 50 and rsi > 50) and volatility_ok:
                sl = close - 2 * atr
                tp = close + 3 * atr
                self.tp = tp
                self.sl = sl
                self.last_entry_price = close
                self.buy(sl=self.sl, tp=self.tp)
        else:
            # Trailing stop dinámico
            new_sl = max(self.sl, close - atr)
            self.sl = new_sl

            # Toma de ganancias parcial
            if close >= self.last_entry_price + 2 * atr:
                tp = self.last_entry_price + 1.5 * atr

            exit_condition = (
                close < self.sl or
                close > self.tp or
                (rsi > 70 and prev_rsi > 70) or
                (ema9 < ema21)
            )

            if exit_condition:
                self.position.close()

class SimpleTradingFitness:
    def __init__(self, data=bk_data, invalid_fitness=1e6):
        self.invalid_fitness = invalid_fitness
        self.data = data

    def evaluate(self, individual):
        # Convert grammar output to Python expression
        try:
            # Example: parse individual into logic/parameter code
            # This depends on how your grammar is defined
            evolved_code = compile("params = " + individual, '<string>', 'exec')
            local_vars = {}
            exec(evolved_code, {}, local_vars)
            params = local_vars["params"]

            # For example, your grammar might produce:
            # {'rsi_entry': 50, 'atr_sl_mult': 2, 'atr_tp_mult': 3}

            # Inject into a modified version of your strategy
            class GE_Strategy(MyStrategy):
                def init(self):
                    super().init()
                    self.rsi_entry = params.get('rsi_entry', 50)
                    self.atr_sl_mult = params.get('atr_sl_mult', 2)
                    self.atr_tp_mult = params.get('atr_tp_mult', 3)

                def next(self):
                    close = self.data.Close[-1]
                    atr = self.atr[-1]
                    rsi = self.rsi[-1]
                    prev_rsi = self.rsi[-2]
                    ema9 = self.ema9[-1]
                    ema21 = self.ema21[-1]
                    ema200 = self.ema200[-1]

                    bullish_trend = close > ema200
                    ema_crossover = ema9 > ema21
                    volatility_ok = atr > 0.0005

                    if not self.position:
                        if bullish_trend and ema_crossover and (30 < rsi < 70) and (prev_rsi < self.rsi_entry and rsi > self.rsi_entry) and volatility_ok:
                            sl = close - self.atr_sl_mult * atr
                            tp = close + self.atr_tp_mult * atr
                            self.tp = tp
                            self.sl = sl
                            self.last_entry_price = close
                            self.buy(sl=self.sl, tp=self.tp)
                    else:
                        new_sl = max(self.sl, close - atr)
                        self.sl = new_sl
                        if close >= self.last_entry_price + 2 * atr:
                            tp = self.last_entry_price + 1.5 * atr

                        exit_condition = (
                            close < self.sl or
                            close > self.tp or
                            (rsi > 70 and prev_rsi > 70) or
                            (ema9 < ema21)
                        )

                        if exit_condition:
                            self.position.close()

            cash = 1_000_000
            bt = Backtest(self.data, GE_Strategy, cash=cash, commission=0.001, exclusive_orders=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats = bt.run()

            #expectancy = stats.get('Expectancy [%]', 0)
            #profit_factor = stats.get('Profit Factor', 0)
            final_equity = stats.get('Equity Final [$]', 0)
            percent_profit = (final_equity - cash) / cash * 100
            #if np.isnan(expectancy) or np.isnan(profit_factor):
            #    return self.invalid_fitness, {
            #        "generation": 0, "evals": 1, "test_error": self.invalid_fitness
            #    }

            #score = -expectancy * profit_factor  # NEGATIVE because GE minimizes fitness
            score = -percent_profit

            return score, {
                "generation": 0,
                "evals": 1,
                "test_error": score
            }

        except Exception as e:
            return self.invalid_fitness, {
                "generation": 0,
                "evals": 1,
                "test_error": self.invalid_fitness
            }


if __name__ == "__main__":
    fitness = SimpleTradingFitness()
    sge.evolutionary_algorithm(evaluation_function=fitness, parameters_file="parameters/standard_trading.yml")
