# Trading Bot MQL5

## Overview
This project is a trading bot implemented in MQL5, designed to automate trading strategies on the MetaTrader 5 platform. The bot utilizes various classes for risk management, order management, and signal processing to execute trades based on market conditions.

## Project Structure
- **Experts/TradingBot.mq5**: The main expert advisor that initializes the bot and manages trading logic.
- **Include/RiskManager.mqh**: Contains the `RiskManager` class for calculating position sizes and managing risk.
- **Include/OrderManager.mqh**: Defines the `OrderManager` class for managing open orders and closing positions.
- **Include/Utils.mqh**: Utility functions for logging and configuration loading.
- **Libraries/SignalProcessor.mqh**: The `SignalProcessor` class for processing trading signals.
- **Scripts/BacktestRunner.mq5**: A script for backtesting the trading bot using historical data.

## Setup Instructions
1. Copy the entire project folder to the `MQL5` directory of your MetaTrader 5 installation.
2. Open MetaEditor and compile all the `.mq5` and `.mqh` files.
3. Attach the `TradingBot` expert advisor to a chart in MetaTrader 5 to start trading.

## Usage Guidelines
- Configure the trading parameters in the `TradingBot.mq5` file before running the bot.
- Monitor the logs for any errors or important information regarding trade execution.
- Use the `BacktestRunner.mq5` script to simulate trades and evaluate the bot's performance on historical data.

## License
This project is open-source and can be modified and distributed under the terms of the MIT License.