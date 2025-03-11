import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from trading_bot import TradingBot
from strategies import MovingAverageCrossover, RSIStrategy, MACDStrategy, CombinedStrategy, PriceActionStrategy

def backtest_strategy(strategy, symbol="EURUSD", timeframe=mt5.TIMEFRAME_H1, days=30):
    """
    Backtest a trading strategy on historical data
    
    Parameters:
    strategy: A Strategy object
    symbol: Trading symbol
    timeframe: MT5 timeframe constant
    days: Number of days for historical data
    
    Returns:
    Performance metrics
    """
    # Initialize MT5 connection
    if not mt5.initialize():
        print(f"initialize() failed, error code = {mt5.last_error()}")
        return None
    
    try:
        # Get historical data
        timezone = pytz.timezone("Etc/UTC")
        to_date = datetime.now(timezone)
        from_date = to_date - timedelta(days=days)
        
        # Get historical data
        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
        if rates is None or len(rates) == 0:
            print(f"Failed to get historical data for {symbol}")
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Apply strategy
        df = strategy.calculate(df)
        if df is None:
            print("Strategy calculation failed")
            return None
        
        # Calculate returns
        df['position'] = df['signal'].shift(1).fillna(0)  # Position based on previous signal
        df['returns'] = df['close'].pct_change() * df['position']
        
        # Calculate performance metrics
        total_return = df['returns'].sum() * 100
        sharpe_ratio = (df['returns'].mean() / df['returns'].std()) * (252 ** 0.5)  # Annualized
        win_rate = len(df[df['returns'] > 0]) / len(df[df['returns'] != 0]) if len(df[df['returns'] != 0]) > 0 else 0
        
        # Print performance summary
        print(f"===== {strategy.name} Backtest Results =====")
        print(f"Symbol: {symbol}, Timeframe: {timeframe}, Period: {days} days")
        print(f"Total Return: {total_return:.2f}%")
        print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"Win Rate: {win_rate:.2f}")
        print("=====================================")
        
        # Plot strategy
        strategy.plot(df)
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot((1 + df['returns']).cumprod(), label='Strategy Returns')
        plt.title(f'{strategy.name} Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.grid(True)
        plt.legend()
        plt.show()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'df': df
        }
    finally:
        mt5.shutdown()

def run_live_bot_with_strategy(strategy_class, **kwargs):
    """
    Run the trading bot live with a specific strategy
    
    Parameters:
    strategy_class: The strategy class to use
    **kwargs: Additional parameters for the trading bot
    """
    # Create a custom trading bot that uses the specified strategy
    class StrategyBot(TradingBot):
        def __init__(self, strategy, **kwargs):
            super().__init__(**kwargs)
            self.strategy = strategy
            
        def calculate_signals(self, df):
            # Use the strategy to calculate signals instead of the default method
            return self.strategy.calculate(df)
    
    # Create the strategy instance
    strategy = strategy_class()
    
    # Create and run the bot with the strategy
    bot = StrategyBot(strategy, **kwargs)
    bot.run()

if __name__ == "__main__":
    print("MT5 Trading Bot Strategy Tester")
    print("===============================")
    
    # Test if MT5 is installed and can be initialized
    if not mt5.initialize():
        print(f"MT5 initialization failed. Error code: {mt5.last_error()}")
        print("Please make sure MetaTrader 5 is installed and running.")
        exit(1)
    
    mt5.shutdown()
    
    print("Available strategies:")
    print("1. Moving Average Crossover")
    print("2. RSI Strategy")
    print("3. MACD Strategy")
    print("4. Combined Strategy")
    print("5. Price Action Strategy")
    print("6. Run Live Trading Bot")
    
    choice = input("Select a strategy to backtest (1-6): ")
    
    symbol = input("Enter symbol (default: EURUSD): ") or "EURUSD"
    days = int(input("Enter backtest period in days (default: 30): ") or "30")
    
    if choice == "1":
        fast = int(input("Enter fast period (default: 20): ") or "20")
        slow = int(input("Enter slow period (default: 50): ") or "50")
        strategy = MovingAverageCrossover(fast_period=fast, slow_period=slow)
        backtest_strategy(strategy, symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=days)
        
    elif choice == "2":
        period = int(input("Enter RSI period (default: 14): ") or "14")
        overbought = int(input("Enter overbought level (default: 70): ") or "70")
        oversold = int(input("Enter oversold level (default: 30): ") or "30")
        strategy = RSIStrategy(period=period, overbought=overbought, oversold=oversold)
        backtest_strategy(strategy, symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=days)
        
    elif choice == "3":
        fast = int(input("Enter fast period (default: 12): ") or "12")
        slow = int(input("Enter slow period (default: 26): ") or "26")
        signal = int(input("Enter signal period (default: 9): ") or "9")
        strategy = MACDStrategy(fast_period=fast, slow_period=slow, signal_period=signal)
        backtest_strategy(strategy, symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=days)
        
    elif choice == "4":
        strategy = CombinedStrategy()
        backtest_strategy(strategy, symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=days)
        
    elif choice == "5":
        lookback = int(input("Enter lookback period (default: 100): ") or "100")
        touches = int(input("Enter minimum touches (default: 2): ") or "2")
        threshold = float(input("Enter price threshold % (default: 2.0): ") or "2.0") / 100
        rr_ratio = float(input("Enter minimum risk/reward ratio (default: 2.0): ") or "2.0")
        strategy = PriceActionStrategy(lookback_period=lookback, min_touches=touches,
                                     price_threshold=threshold, min_rr_ratio=rr_ratio)
        backtest_strategy(strategy, symbol=symbol, timeframe=mt5.TIMEFRAME_H1, days=days)
        
    elif choice == "6":
        print("Available strategies for live trading:")
        print("1. Moving Average Crossover")
        print("2. RSI Strategy")
        print("3. MACD Strategy")
        print("4. Combined Strategy")
        print("5. Price Action Strategy")
        
        strat_choice = input("Select a strategy for live trading (1-5): ")
        lot_size = float(input("Enter lot size (default: 0.01): ") or "0.01")
        
        if strat_choice == "1":
            run_live_bot_with_strategy(MovingAverageCrossover, 
                                     symbol=symbol, 
                                     lot_size=lot_size)
        elif strat_choice == "2":
            run_live_bot_with_strategy(RSIStrategy,
                                     symbol=symbol, 
                                     lot_size=lot_size)
        elif strat_choice == "3":
            run_live_bot_with_strategy(MACDStrategy,
                                     symbol=symbol, 
                                     lot_size=lot_size)
        elif strat_choice == "4":
            run_live_bot_with_strategy(CombinedStrategy,
                                     symbol=symbol, 
                                     lot_size=lot_size)
        elif strat_choice == "5":
            run_live_bot_with_strategy(PriceActionStrategy,
                                     symbol=symbol, 
                                     lot_size=lot_size)
        else:
            print("Invalid choice")
    
    else:
        print("Invalid choice")