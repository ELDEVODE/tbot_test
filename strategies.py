import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Strategy:
    """Base strategy class for trading signals"""
    def __init__(self, name="BaseStrategy"):
        self.name = name
        
    def calculate(self, df):
        """Calculate signals based on the strategy logic"""
        return df
        
    def plot(self, df):
        """Plot the strategy indicators and signals"""
        pass


class MovingAverageCrossover(Strategy):
    """Moving Average Crossover strategy"""
    def __init__(self, fast_period=20, slow_period=50):
        super().__init__("MA Crossover")
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def calculate(self, df):
        """Calculate MA crossover signals"""
        if df is None or len(df) < self.slow_period:
            return None
            
        # Calculate moving averages
        df[f'MA{self.fast_period}'] = df['close'].rolling(window=self.fast_period).mean()
        df[f'MA{self.slow_period}'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Generate signals (1 for buy, -1 for sell, 0 for no action)
        df['signal'] = 0
        df.loc[df[f'MA{self.fast_period}'] > df[f'MA{self.slow_period}'], 'signal'] = 1
        df.loc[df[f'MA{self.fast_period}'] < df[f'MA{self.slow_period}'], 'signal'] = -1
        
        return df
    
    def plot(self, df):
        """Plot MA crossover"""
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['close'], label='Close Price')
        plt.plot(df.index, df[f'MA{self.fast_period}'], label=f'MA{self.fast_period}')
        plt.plot(df.index, df[f'MA{self.slow_period}'], label=f'MA{self.slow_period}')
        
        # Plot buy signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        plt.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='g', s=100, label='Buy')
        plt.scatter(sell_signals.index, sell_signals['close'], 
                   marker='v', color='r', s=100, label='Sell')
                   
        plt.title(f'{self.name} Strategy')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()


class RSIStrategy(Strategy):
    """Relative Strength Index (RSI) strategy"""
    def __init__(self, period=14, overbought=70, oversold=30):
        super().__init__("RSI Strategy")
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
        
    def calculate(self, df):
        """Calculate RSI signals"""
        if df is None or len(df) < self.period + 10:
            return None
            
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['RSI'] < self.oversold, 'signal'] = 1  # Buy when RSI below oversold
        df.loc[df['RSI'] > self.overbought, 'signal'] = -1  # Sell when RSI above overbought
        
        return df
    
    def plot(self, df):
        """Plot RSI indicator and signals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart
        ax1.plot(df.index, df['close'], label='Close Price')
        
        # Plot buy signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   marker='v', color='r', s=100, label='Sell')
        
        ax1.set_title(f'{self.name}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()
        
        # Plot RSI
        ax2.plot(df.index, df['RSI'], color='purple', label='RSI')
        ax2.axhline(y=self.overbought, color='r', linestyle='--', label=f'Overbought ({self.overbought})')
        ax2.axhline(y=self.oversold, color='g', linestyle='--', label=f'Oversold ({self.oversold})')
        ax2.axhline(y=50, color='k', linestyle='-')
        ax2.set_ylabel('RSI')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()
        plt.show()


class MACDStrategy(Strategy):
    """Moving Average Convergence Divergence (MACD) strategy"""
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        super().__init__("MACD Strategy")
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        
    def calculate(self, df):
        """Calculate MACD signals"""
        if df is None or len(df) < self.slow_period + self.signal_period:
            return None
            
        # Calculate MACD
        df['EMA_fast'] = df['close'].ewm(span=self.fast_period, adjust=False).mean()
        df['EMA_slow'] = df['close'].ewm(span=self.slow_period, adjust=False).mean()
        df['MACD'] = df['EMA_fast'] - df['EMA_slow']
        df['MACD_signal'] = df['MACD'].ewm(span=self.signal_period, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Generate signals
        df['signal'] = 0
        # Buy when MACD crosses above signal line
        df.loc[(df['MACD'] > df['MACD_signal']) & 
               (df['MACD'].shift(1) <= df['MACD_signal'].shift(1)), 'signal'] = 1
        
        # Sell when MACD crosses below signal line
        df.loc[(df['MACD'] < df['MACD_signal']) & 
               (df['MACD'].shift(1) >= df['MACD_signal'].shift(1)), 'signal'] = -1
        
        return df
    
    def plot(self, df):
        """Plot MACD indicator and signals"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price chart
        ax1.plot(df.index, df['close'], label='Close Price')
        
        # Plot buy signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['close'], 
                   marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['close'], 
                   marker='v', color='r', s=100, label='Sell')
        
        ax1.set_title(f'{self.name}')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid()
        
        # Plot MACD
        ax2.plot(df.index, df['MACD'], color='blue', label='MACD')
        ax2.plot(df.index, df['MACD_signal'], color='red', label='Signal')
        
        # Plot histogram
        pos = df[df['MACD_hist'] > 0]
        neg = df[df['MACD_hist'] < 0]
        ax2.bar(pos.index, pos['MACD_hist'], color='green', alpha=0.5, width=0.8)
        ax2.bar(neg.index, neg['MACD_hist'], color='red', alpha=0.5, width=0.8)
        
        ax2.set_ylabel('MACD')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid()
        
        plt.tight_layout()
        plt.show()


class CombinedStrategy(Strategy):
    """Combined strategy using multiple indicators"""
    def __init__(self):
        super().__init__("Combined Strategy")
        self.rsi = RSIStrategy(period=14, overbought=70, oversold=30)
        self.macd = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
        self.ma = MovingAverageCrossover(fast_period=20, slow_period=50)
        
    def calculate(self, df):
        """Calculate combined strategy signals"""
        if df is None or len(df) < 50:
            return None
            
        # Calculate individual strategy signals
        df = self.rsi.calculate(df)
        df = self.macd.calculate(df)
        df = self.ma.calculate(df)
        
        # Rename signals
        df['rsi_signal'] = df['signal']
        df = df.drop('signal', axis=1)
        df = self.macd.calculate(df)
        df['macd_signal'] = df['signal']
        df = df.drop('signal', axis=1)
        df = self.ma.calculate(df)
        df['ma_signal'] = df['signal']
        
        # Generate combined signal (majority vote)
        df['signal_sum'] = df['rsi_signal'] + df['macd_signal'] + df['ma_signal']
        df['signal'] = 0
        df.loc[df['signal_sum'] >= 2, 'signal'] = 1  # Buy if at least 2 buy signals
        df.loc[df['signal_sum'] <= -2, 'signal'] = -1  # Sell if at least 2 sell signals
        
        return df


class PriceActionStrategy(Strategy):
    """Price Action strategy based on support/resistance levels"""
    def __init__(self, lookback_period=100, min_touches=2, price_threshold=0.02, min_rr_ratio=2):
        super().__init__("Price Action Strategy")
        self.lookback_period = lookback_period
        self.min_touches = min_touches
        self.price_threshold = price_threshold  # % threshold for price levels
        self.min_rr_ratio = min_rr_ratio
        
    def find_support_resistance(self, df):
        """Find support and resistance levels based on historical price action"""
        levels = []
        
        # Get local maxima and minima
        df['Local Max'] = df['high'].rolling(window=20, center=True).apply(lambda x: x[10] == max(x))
        df['Local Min'] = df['low'].rolling(window=20, center=True).apply(lambda x: x[10] == min(x))
        
        # Identify potential levels from local extremes
        for i in range(len(df)-1):
            if df['Local Max'].iloc[i]:
                level = df['high'].iloc[i]
                touches = 0
                # Count touches of this level
                for j in range(max(0, i-self.lookback_period), min(len(df), i+self.lookback_period)):
                    if abs(df['high'].iloc[j] - level) <= level * self.price_threshold:
                        touches += 1
                if touches >= self.min_touches:
                    levels.append({'price': level, 'type': 'resistance'})
                    
            if df['Local Min'].iloc[i]:
                level = df['low'].iloc[i]
                touches = 0
                # Count touches of this level
                for j in range(max(0, i-self.lookback_period), min(len(df), i+self.lookback_period)):
                    if abs(df['low'].iloc[j] - level) <= level * self.price_threshold:
                        touches += 1
                if touches >= self.min_touches:
                    levels.append({'price': level, 'type': 'support'})
        
        return levels
    
    def identify_candlestick_patterns(self, df):
        """Identify bullish and bearish candlestick patterns"""
        df['body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        
        # Pin bars
        df['pin_bar_up'] = (df['lower_wick'] > df['body'].abs() * 2) & (df['upper_wick'] < df['body'].abs())
        df['pin_bar_down'] = (df['upper_wick'] > df['body'].abs() * 2) & (df['lower_wick'] < df['body'].abs())
        
        # Engulfing patterns
        df['bullish_engulfing'] = (df['body'] > 0) & (df['body'].shift(1) < 0) & \
                                 (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        df['bearish_engulfing'] = (df['body'] < 0) & (df['body'].shift(1) > 0) & \
                                 (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
        return df
    
    def calculate(self, df):
        """Calculate trading signals based on price action near S/R levels"""
        if df is None or len(df) < self.lookback_period:
            return None
            
        # Find support and resistance levels
        levels = self.find_support_resistance(df)
        
        # Identify candlestick patterns
        df = self.identify_candlestick_patterns(df)
        
        # Initialize signal column
        df['signal'] = 0
        df['level_type'] = ''
        df['risk_reward'] = 0
        
        # Generate trading signals
        for i in range(1, len(df)):
            current_price = df['close'].iloc[i]
            
            # Check each level for potential trades
            for level in levels:
                price_distance = abs(current_price - level['price'])
                
                # Price near level (within threshold)
                if price_distance <= current_price * self.price_threshold:
                    df['level_type'].iloc[i] = level['type']
                    
                    # Bullish setup at support
                    if level['type'] == 'support' and \
                       (df['pin_bar_up'].iloc[i] or df['bullish_engulfing'].iloc[i]):
                        # Calculate potential risk/reward
                        stop_loss = df['low'].iloc[i] - (price_distance * 0.5)  # Stop below support
                        # Target the nearest resistance or 1:self.min_rr_ratio risk/reward
                        take_profit = current_price + (price_distance * self.min_rr_ratio)
                        risk_reward = abs(take_profit - current_price) / abs(stop_loss - current_price)
                        
                        if risk_reward >= self.min_rr_ratio:
                            df['signal'].iloc[i] = 1
                            df['risk_reward'].iloc[i] = risk_reward
                    
                    # Bearish setup at resistance
                    elif level['type'] == 'resistance' and \
                         (df['pin_bar_down'].iloc[i] or df['bearish_engulfing'].iloc[i]):
                        # Calculate potential risk/reward
                        stop_loss = df['high'].iloc[i] + (price_distance * 0.5)  # Stop above resistance
                        # Target the nearest support or 1:self.min_rr_ratio risk/reward
                        take_profit = current_price - (price_distance * self.min_rr_ratio)
                        risk_reward = abs(take_profit - current_price) / abs(stop_loss - current_price)
                        
                        if risk_reward >= self.min_rr_ratio:
                            df['signal'].iloc[i] = -1
                            df['risk_reward'].iloc[i] = risk_reward
        
        return df
    
    def plot(self, df):
        """Plot price action with support/resistance levels and signals"""
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot candlesticks
        for i in range(len(df)):
            if df['close'].iloc[i] > df['open'].iloc[i]:
                color = 'g'
            else:
                color = 'r'
            
            # Plot candlestick body
            ax1.plot([i, i], [df['open'].iloc[i], df['close'].iloc[i]], 
                    color=color, linewidth=3)
            # Plot wicks
            ax1.plot([i, i], [df['low'].iloc[i], df['high'].iloc[i]], 
                    color=color, linewidth=1)
        
        # Plot support and resistance levels
        levels = self.find_support_resistance(df)
        for level in levels:
            if level['type'] == 'support':
                ax1.axhline(y=level['price'], color='g', linestyle='--', alpha=0.5)
            else:
                ax1.axhline(y=level['price'], color='r', linestyle='--', alpha=0.5)
        
        # Plot signals
        buy_signals = df[df['signal'] == 1]
        sell_signals = df[df['signal'] == -1]
        
        ax1.scatter(buy_signals.index, buy_signals['low'] * 0.999, 
                   marker='^', color='g', s=100, label='Buy')
        ax1.scatter(sell_signals.index, sell_signals['high'] * 1.001, 
                   marker='v', color='r', s=100, label='Sell')
        
        ax1.set_title(f'{self.name} - Support/Resistance Levels')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        plt.tight_layout()
        plt.show()