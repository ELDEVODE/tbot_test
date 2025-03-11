//+------------------------------------------------------------------+
//|                                                      TradingBot.mq5|
//|                        Copyright 2023, MetaQuotes Ltd.             |
//|                                             https://www.mql5.com   |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict

// Include required files
#include <Trade\Trade.mqh>
#include "..\Include\RiskManager.mqh"
#include "..\Include\OrderManager.mqh"
#include "..\Include\Utils.mqh"
#include "..\Libraries\SignalProcessor.mqh"

// Input parameters
input string   InpSymbol = "EURUSD";    // Trading symbol
input double   InpRiskPerTrade = 0.02;   // Risk per trade (%)
input int      InpMaxTrades = 3;         // Maximum concurrent trades
input double   InpLotSize = 0.1;         // Default lot size

// Global variables
CTrade trade;
RiskManager *riskManager;
OrderManager *orderManager;
SignalProcessor *signalProcessor;

//+------------------------------------------------------------------+
//| Expert initialization function                                     |
//+------------------------------------------------------------------+
int OnInit()
{
    // Initialize objects
    riskManager = new RiskManager(InpRiskPerTrade, InpMaxTrades);
    orderManager = new OrderManager(InpSymbol);
    signalProcessor = new SignalProcessor();
    
    // Set trading symbol
    if(!SymbolSelect(InpSymbol, true)) {
        Print("Error: Symbol ", InpSymbol, " not found!");
        return INIT_FAILED;
    }
    
    Print("Trading Bot Initialized for ", InpSymbol);
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                   |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
    // Clean up objects
    delete riskManager;
    delete orderManager;
    delete signalProcessor;
    
    Print("Trading Bot stopped. Reason code: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    // Get current prices
    MqlTick latest_prices;
    if(!SymbolInfoTick(Symbol(), latest_prices)) {
        Print("Error getting latest prices!");
        return;
    }
    
    // Calculate signal
    double maShort = iMA(Symbol(), PERIOD_CURRENT, 20, 0, MODE_SMA, PRICE_CLOSE);
    double maLong = iMA(Symbol(), PERIOD_CURRENT, 50, 0, MODE_SMA, PRICE_CLOSE);
    double rsi = iRSI(Symbol(), PERIOD_CURRENT, 14, PRICE_CLOSE);
    
    double signal = signalProcessor.CalculateSignal(maShort, maLong, rsi);
    
    // Place orders based on signal
    if(MathAbs(signal) > 0.5) {
        double lotSize = riskManager.CalculatePositionSize(
            AccountInfoDouble(ACCOUNT_BALANCE),
            latest_prices.ask,
            latest_prices.ask * 0.99, // Example stop loss at 1%
            SymbolInfoDouble(Symbol(), SYMBOL_TRADE_TICK_VALUE)
        );
        
        if(signal > 0) {
            trade.Buy(lotSize, Symbol(), 0, 0, 0, "Buy Signal");
        }
        else if(signal < 0) {
            trade.Sell(lotSize, Symbol(), 0, 0, 0, "Sell Signal");
        }
    }
    
    // Update open positions
    orderManager.UpdateOpenOrders();
}