//+------------------------------------------------------------------+
//|                                                   RiskManager.mqh |
//+------------------------------------------------------------------+
class RiskManager
{
private:
    double m_maxRiskPerTrade;
    int    m_maxTrades;

public:
    // Constructor
    RiskManager(double maxRisk = 0.02, int maxTradesAllowed = 3)
    {
        m_maxRiskPerTrade = maxRisk;
        m_maxTrades = maxTradesAllowed;
    }

    // Method to calculate position size
    double CalculatePositionSize(
        double accountBalance,
        double entryPrice,
        double stopLoss,
        double tickValue
    )
    {
        if(stopLoss == 0.0)
            return 0.01; // Minimum lot size if no stop loss

        // Calculate risk amount with safety factor
        double riskAmount = accountBalance * m_maxRiskPerTrade * 0.8;
        
        double stopLossPips = MathAbs(entryPrice - stopLoss) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
        double positionSize = riskAmount / (stopLossPips * tickValue);
        
        // Conservative position sizing
        double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
        double maxLot = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), 0.5);
        double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        positionSize = MathFloor(positionSize / lotStep) * lotStep;
        positionSize = MathMax(minLot, MathMin(positionSize, maxLot));
        
        return positionSize;
    }
};