import yfinance as yf
import pandas as pd

def check_yfinance_capabilities():
    # Pick a large cap Indian stock for testing
    ticker_symbol = "RELIANCE.NS" 
    ticker = yf.Ticker(ticker_symbol)
    
    print(f"Checking data for {ticker_symbol}...")
    
    # 1. Check Minute Level Data
    print("\n--- Minute Level Data ---")
    try:
        # Try to fetch 1m data for max period allowed
        # yfinance usually limits 1m data to last 7 days, let's see
        hist_1m = ticker.history(period="max", interval="1m")
        print(f"1m data shape: {hist_1m.shape}")
        if not hist_1m.empty:
            print(f"1m data range: {hist_1m.index.min()} to {hist_1m.index.max()}")
        else:
            print("No 1m data found with period='max'.")
            
        # Try specific recent range
        hist_1m_recent = ticker.history(period="5d", interval="1m")
        print(f"1m data (5d) shape: {hist_1m_recent.shape}")
        
    except Exception as e:
        print(f"Error fetching minute data: {e}")

    # 2. Check Earnings/Calendar Data
    print("\n--- Earnings/Calendar Data ---")
    try:
        calendar = ticker.calendar
        print("Calendar:")
        print(calendar)
        
        earnings_dates = ticker.earnings_dates
        print("\nEarnings Dates:")
        if earnings_dates is not None:
            print(earnings_dates.head())
        else:
            print("No earnings_dates found.")
            
    except Exception as e:
        print(f"Error fetching calendar data: {e}")

if __name__ == "__main__":
    check_yfinance_capabilities()
