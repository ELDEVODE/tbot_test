import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import pytz
from typing import Optional, Dict, List, Union
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)

class RiskManager:
    def __init__(self, max_risk_per_trade: float = 0.01, max_trades: int = 2):
        """
        Initialize risk management parameters
        
        Args:
            max_risk_per_trade (float): Maximum risk per trade as decimal (e.g., 0.02 = 2%)
            max_trades (int): Maximum number of concurrent trades
        """
        self.max_risk_per_trade = max_risk_per_trade
        self.max_trades = max_trades
    
    def calculate_position_size(self, account_info: dict, entry_price: float, stop_loss: float, symbol_info: dict) -> float:
        """
        Calculate position size with enhanced risk controls
        """
        if stop_loss == 0:
            return 0.01  # Minimum lot size if no stop loss
            
        # Calculate risk amount with additional safety factor
        risk_amount = min(account_info['balance'] * self.max_risk_per_trade * 0.7, 1.0)  # Max $1 risk per trade
        
        pip_value = symbol_info['trade_tick_value'] * (symbol_info['point'] / symbol_info['trade_tick_size'])
        stop_loss_pips = abs(entry_price - stop_loss) / symbol_info['point']
        
        position_size = risk_amount / (stop_loss_pips * pip_value)
        
        # More conservative position sizing
        min_lot = symbol_info['volume_min']
        max_lot = min(symbol_info['volume_max'], 0.1)  # Added max lot cap
        step = symbol_info['volume_step']
        
        position_size = round(position_size / step) * step
        position_size = max(min_lot, min(position_size, max_lot))
        
        return position_size

class OrderManager:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.open_orders: List[dict] = []
        
    def update_open_orders(self):
        """Update the list of open orders"""
        positions = mt5.positions_get(symbol=self.symbol)
        self.open_orders = list(positions) if positions else []
        
    def close_all_positions(self) -> bool:
        """Close all open positions for the symbol"""
        success = True
        for position in self.open_orders:
            order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = mt5.symbol_info_tick(self.symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(self.symbol).ask
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": position.volume,
                "type": order_type,
                "position": position.ticket,
                "price": close_price,
                "deviation": 20,
                "magic": 234000,
                "comment": "Close position",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Failed to close position {position.ticket}: {result.comment}")
                success = False
                
        return success

class TradingBot:
    def __init__(self, 
                 symbol: str = "EURUSD",
                 timeframes: List[int] = [mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4],
                 risk_per_trade: float = 0.02,
                 max_trades: int = 3):
        """
        Initialize the trading bot with enhanced parameters
        
        Args:
            symbol (str): Trading symbol
            timeframes (List[int]): List of timeframes to analyze
            risk_per_trade (float): Maximum risk per trade as decimal
            max_trades (int): Maximum number of concurrent trades
        """
        self.symbol = symbol
        self.timeframes = timeframes
        self.risk_manager = RiskManager(risk_per_trade, max_trades)
        self.order_manager = OrderManager(symbol)
        self.initialized = False
        self.timezone = pytz.timezone("Etc/UTC")
        
        # Load configuration if exists
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """Load configuration with more conservative defaults"""
        config_file = 'trading_config.json'
        default_config = {
            'take_profit_multiplier': 1.5,     # Reduced from 2.0
            'trailing_stop': True,
            'trailing_stop_activation': 0.6,    # Increased from 0.5
            'max_spread': 10,                  # Reduced from 20
            'trading_hours': {
                'start': '10:00',              # Later start
                'end': '16:00'                 # Earlier end
            }
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return {**default_config, **json.load(f)}
            except:
                logging.warning(f"Failed to load config file, using defaults")
                
        return default_config
    
    def initialize(self) -> bool:
        """Initialize connection to MT5 terminal with enhanced error checking"""
        if not mt5.initialize():
            logging.error(f"MT5 initialization failed, error code = {mt5.last_error()}")
            return False
            
        # Check symbol and add to MarketWatch if needed
        symbol_info = mt5.symbol_info(self.symbol)
        if (symbol_info is None):
            logging.error(f"Symbol {self.symbol} not found")
            mt5.shutdown()
            return False
            
        if not symbol_info.visible:
            if not mt5.symbol_select(self.symbol, True):
                logging.error(f"Failed to add {self.symbol} to MarketWatch")
                mt5.shutdown()
                return False
                
        account_info = mt5.account_info()
        if account_info is None:
            logging.error("Failed to get account info")
            mt5.shutdown()
            return False
            
        logging.info(f"MT5 connected: {mt5.terminal_info()}")
        logging.info(f"Account: {account_info.login} ({account_info.company})")
        logging.info(f"Balance: {account_info.balance} {account_info.currency}")
        
        self.initialized = True
        return True
    
    def get_historical_data(self, timeframe: int, bars: int = 1000) -> Optional[pd.DataFrame]:
        """Get historical data with error handling"""
        if not self.initialized:
            logging.error("MT5 not initialized")
            return None
            
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, bars)
            if rates is None or len(rates) == 0:
                logging.error(f"Failed to get historical data for {self.symbol}")
                return None
                
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
            
        except Exception as e:
            logging.error(f"Error getting historical data: {e}")
            return None
    
    def calculate_signals(self, dfs: Dict[int, pd.DataFrame]) -> Dict[str, float]:
        """
        Calculate trading signals using multiple timeframe analysis
        Returns dict with signal strength and suggested entry/exit points
        """
        signals = {'strength': 0, 'entry': 0, 'stop_loss': 0, 'take_profit': 0}
        
        try:
            # Process each timeframe
            for timeframe, df in dfs.items():
                if df is None or len(df) < 50:
                    continue
                    
                # Calculate technical indicators
                df['MA20'] = df['close'].rolling(window=20).mean()
                df['MA50'] = df['close'].rolling(window=50).mean()
                df['RSI'] = self.calculate_rsi(df['close'])
                
                # Generate signals per timeframe
                trend = 1 if df['MA20'].iloc[-1] > df['MA50'].iloc[-1] else -1
                momentum = 1 if df['RSI'].iloc[-1] > 50 else -1
                
                # Weight signals by timeframe (higher timeframes have more weight)
                weight = {
                    mt5.TIMEFRAME_M15: 0.2,
                    mt5.TIMEFRAME_H1: 0.3,
                    mt5.TIMEFRAME_H4: 0.5
                }.get(timeframe, 0.3)
                
                signals['strength'] += (trend * momentum * weight)
            
            # Normalize signal strength to [-1, 1] range
            signals['strength'] = max(min(signals['strength'], 1), -1)
            
            # Calculate entry, stop loss and take profit levels
            if abs(signals['strength']) >= 0.5:  # Minimum signal threshold
                current_price = mt5.symbol_info_tick(self.symbol).ask
                atr = self.calculate_atr(dfs[self.timeframes[0]])  # Use shortest timeframe for ATR
                
                signals['entry'] = current_price
                signals['stop_loss'] = current_price - (atr * 1.5) if signals['strength'] > 0 else current_price + (atr * 1.5)
                tp_distance = abs(current_price - signals['stop_loss']) * self.config['take_profit_multiplier']
                signals['take_profit'] = current_price + tp_distance if signals['strength'] > 0 else current_price - tp_distance
                
        except Exception as e:
            logging.error(f"Error calculating signals: {e}")
            
        return signals
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]
    
    def place_order(self, signals: Dict[str, float]) -> Optional[dict]:
        """Place a trade with proper risk management"""
        if not self.initialized:
            logging.error("MT5 not initialized")
            return None
            
        # Update open orders
        self.order_manager.update_open_orders()
        if len(self.order_manager.open_orders) >= self.risk_manager.max_trades:
            logging.info("Maximum number of trades reached")
            return None
            
        # Check trading hours
        current_time = datetime.now().strftime('%H:%M')
        if not (self.config['trading_hours']['start'] <= current_time <= self.config['trading_hours']['end']):
            logging.info("Outside trading hours")
            return None
            
        # Check spread
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info.spread > self.config['max_spread']:
            logging.info(f"Spread too high: {symbol_info.spread} points")
            return None
            
        # Determine order type based on signal strength
        if abs(signals['strength']) < 0.5:
            return None
            
        order_type = mt5.ORDER_TYPE_BUY if signals['strength'] > 0 else mt5.ORDER_TYPE_SELL
        
        # Calculate position size
        account_info = mt5.account_info()._asdict()
        lot_size = self.risk_manager.calculate_position_size(
            account_info,
            signals['entry'],
            signals['stop_loss'],
            symbol_info._asdict()
        )
        
        if lot_size == 0:
            logging.warning("Position size calculation failed")
            return None
            
        # Prepare the order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": order_type,
            "price": signals['entry'],
            "sl": signals['stop_loss'],
            "tp": signals['take_profit'],
            "deviation": 20,
            "magic": 234000,
            "comment": f"Signal strength: {signals['strength']:.2f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,
        }
        
        # Send the order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Order failed: {result.comment}")
            return None
            
        logging.info(f"Order placed successfully: {result.order}")
        return result._asdict()
    
    def update_trailing_stops(self):
        """Update trailing stops for open positions"""
        if not self.config['trailing_stop']:
            return
            
        self.order_manager.update_open_orders()
        for position in self.order_manager.open_orders:
            if position.tp == 0:
                continue
                
            profit_distance = abs(position.tp - position.price_open)
            activation_level = position.price_open + (profit_distance * self.config['trailing_stop_activation'])
            
            current_price = mt5.symbol_info_tick(self.symbol).bid
            if position.type == mt5.ORDER_TYPE_BUY and current_price >= activation_level:
                new_sl = current_price - (profit_distance * 0.5)
                if new_sl > position.sl:
                    self.modify_sl_tp(position.ticket, new_sl, position.tp)
            elif position.type == mt5.ORDER_TYPE_SELL and current_price <= activation_level:
                new_sl = current_price + (profit_distance * 0.5)
                if new_sl < position.sl:
                    self.modify_sl_tp(position.ticket, new_sl, position.tp)
    
    def modify_sl_tp(self, ticket: int, sl: float, tp: float) -> bool:
        """Modify stop loss and take profit levels for an open position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logging.error(f"Failed to modify SL/TP: {result.comment}")
            return False
        return True
    
    def run(self):
        """Main bot execution loop with enhanced error handling and monitoring"""
        if not self.initialize():
            return
            
        logging.info(f"Starting trading bot for {self.symbol}...")
        
        try:
            while True:
                try:
                    # Get data for all timeframes
                    dfs = {timeframe: self.get_historical_data(timeframe) 
                          for timeframe in self.timeframes}
                    
                    if any(df is None for df in dfs.values()):
                        time.sleep(60)
                        continue
                    
                    # Calculate signals
                    signals = self.calculate_signals(dfs)
                    
                    # Place new orders if conditions are met
                    if abs(signals['strength']) >= 0.5:
                        self.place_order(signals)
                    
                    # Update trailing stops
                    self.update_trailing_stops()
                    
                    # Log current status
                    self.order_manager.update_open_orders()
                    logging.info(f"Active trades: {len(self.order_manager.open_orders)}, Signal strength: {signals['strength']:.2f}")
                    
                except Exception as e:
                    logging.error(f"Error in main loop: {e}")
                    
                # Wait before next iteration
                time.sleep(60)
                
        except KeyboardInterrupt:
            logging.info("Bot stopped manually")
        except Exception as e:
            logging.error(f"Critical error: {e}")
        finally:
            # Clean up
            if self.initialized:
                self.order_manager.close_all_positions()
                mt5.shutdown()
                logging.info("Bot shutdown complete")


if __name__ == "__main__":
    # Create and start the bot with default parameters
    bot = TradingBot(
        symbol="EURUSD",
        timeframes=[mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4],
        risk_per_trade=0.001,
        max_trades=2
    )
    
    # Run the bot
    bot.run()