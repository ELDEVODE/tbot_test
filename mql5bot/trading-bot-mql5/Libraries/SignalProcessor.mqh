//+------------------------------------------------------------------+
//|                                               SignalProcessor.mqh |
//+------------------------------------------------------------------+
class SignalProcessor
{
public:
    // Method to calculate trading signals based on market data
    double CalculateSignal(double maShort, double maLong, double rsi)
    {
        double signal = 0.0;
        
        // Buy if short MA is above long MA and RSI is below 30
        if(maShort > maLong && rsi < 30)
        {
            signal = 1.0;
        }
        // Sell if short MA is below long MA and RSI is above 70
        else if(maShort < maLong && rsi > 70)
        {
            signal = -1.0;
        }
        
        return signal;
    }
    
    // Method to calculate moving average
    double CalculateMA(double &prices[], int period)
    {
        double sum = 0.0;
        int count = MathMin(ArraySize(prices), period);
        
        for (int i = 0; i < count; i++)
        {
            sum += prices[i];
        }
        
        return sum / count;
    }

    // Method to calculate RSI
    double CalculateRSI(double &prices[], int period)
    {
        double gain = 0.0;
        double loss = 0.0;
        
        for (int i = 1; i < period; i++)
        {
            double change = prices[i] - prices[i - 1];
            if (change > 0)
            {
                gain += change;
            }
            else
            {
                loss -= change;
            }
        }
        
        gain /= period;
        loss /= period;

        if (loss == 0) return 100; // Avoid division by zero

        double rs = gain / loss;
        return 100 - (100 / (1 + rs));
    }
};