//+------------------------------------------------------------------+
//|                                                   OrderManager.mqh |
//+------------------------------------------------------------------+
#include <Trade\Trade.mqh>

class OrderManager
{
private:
    string   m_symbol;
    CTrade   m_trade;
    ulong    m_positions[];

public:
    OrderManager(string symbol)
    {
        m_symbol = symbol;
        m_trade.SetExpertMagicNumber(234000);
    }

    void UpdateOpenOrders()
    {
        ArrayResize(m_positions, 0);
        int total = PositionsTotal();
        
        for(int i = 0; i < total; i++)
        {
            ulong ticket = PositionGetTicket(i);
            if(ticket <= 0) continue;
            
            if(PositionGetString(POSITION_SYMBOL) == m_symbol)
            {
                int size = ArraySize(m_positions);
                ArrayResize(m_positions, size + 1);
                m_positions[size] = ticket;
            }
        }
    }

    bool CloseAllPositions()
    {
        bool success = true;
        UpdateOpenOrders();
        
        for(int i = 0; i < ArraySize(m_positions); i++)
        {
            if(!m_trade.PositionClose(m_positions[i]))
            {
                Print("Failed to close position ", m_positions[i], ": ", m_trade.ResultRetcode());
                success = false;
            }
        }
        return success;
    }
};