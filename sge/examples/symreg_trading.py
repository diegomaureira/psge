from sge.utilities.protected_math import _log_, _div_, _exp_, _inv_, _sqrt_, protdiv
import numpy as np
from numpy import cos, sin, arange, random
import pandas as pd
from binance.client import Client
import dotenv
import os
from sklearn.model_selection import train_test_split
import ta
from backtesting import Backtest, Strategy
import warnings

def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step

class Trading():
    def __init__(self, has_test_set=True, invalid_fitness=9999999, cash=1_000_000):
        self.__train_set = []
        self.__test_set = None
        self.__invalid_fitness = invalid_fitness
        self.partition_rng = random.uniform()
        self.function = function
        self.has_test_set = has_test_set
        self.cash = cash
        self.read_dataset()

    def read_dataset(self):

        # Load environment variables from .env file
        dotenv.load_dotenv()

        version = 'demo'

        if version == 'demo':
            # Initialize API (replace with your keys)
            api_key = os.getenv('BINANCE_API_TEST_KEY')
            api_secret = os.getenv('BINANCE_API_TEST_SECRET')
            #url = 'https://testnet.binance.vision'
            testnet = True
        elif version == 'real':
            # Initialize API (replace with your keys)
            api_key = os.getenv('BINANCE_API_KEY')
            api_secret = os.getenv('BINANCE_API_SECRET')
            #url = 'https://api.binance.com'
            testnet = False

        # API key/secret are required for user data endpoints
        client = Client(api_key=api_key, api_secret=api_secret, testnet=testnet)

        klines = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1MINUTE, "1 day ago UTC")

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
        #bk_data.index = pd.to_datetime(data['timestamp'], unit='ms')
        # Split data intro train and test with sklearn
        train_data, test_data = train_test_split(bk_data, test_size=0.2, shuffle=False)
        self.__train_set = train_data.values
        self.__test_set = test_data.values
        self.__train_set = pd.DataFrame(self.__train_set, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.__train_set.index = pd.to_datetime(data['timestamp'][1:], unit='ms')
        self.__test_set = pd.DataFrame(self.__test_set, columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        self.__test_set.index = pd.to_datetime(data['timestamp'][1:], unit='ms')

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

    def get_error(self, individual, dataset):
        try:
            bt = Backtest(dataset, individual, cash=self.cash, commission=0.001, exclusive_orders=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stats = bt.run()
            final_equity = stats.get('Equity Final [$]', 0)
            percent_profit = (final_equity - self.cash) / self.cash * 100
            pred_error = -percent_profit
        except (SyntaxError, ValueError, OverflowError, MemoryError, FloatingPointError, ZeroDivisionError):
            pred_error = self.__invalid_fitness
        return pred_error

    def evaluate(self, individual):
        error = 0.0
        test_error = 0.0
        if individual is None:
            return None

        error = self.get_error(individual, self.__train_set)

        if error is None:
            error = self.__invalid_fitness
            
        if self.__test_set is not None:
            test_error = 0
            test_error = self.get_error(individual, self.__test_set)
        total_error = error+test_error

        return total_error, {'generation': 0, "evals": 1, "train_error":error, "test_error": test_error, "total_error": total_error}


if __name__ == "__main__":
    import sge
    eval_func = Trading()
    sge.evolutionary_algorithm(evaluation_function=eval_func, parameters_file="parameters/standard_trading.yml")
