import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')

def fetch_stock_data(ticker_symbol, period="7d", interval="1m", refresh_data=False):
    """
    Fetch historical data for a stock from yfinance.
    
    Args:
        ticker_symbol (str): Stock ticker symbol (e.g., 'RELIANCE.NS').
        period (str): Data period to download (default '7d' for max minute data).
        interval (str): Data interval (default '1m').
        refresh_data (bool): If True, force download from yfinance. If False, try loading from CSV.
        
    Returns:
        pd.DataFrame: DataFrame with historical data.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    filename = f"{ticker_symbol}_{period}_{interval}.csv"
    filepath = os.path.join(RAW_DIR, filename)
    
    if not refresh_data and os.path.exists(filepath):
        logger.info(f"Loading cached data for {ticker_symbol} from {filepath}")
        try:
            df = pd.read_csv(filepath)
            # Ensure Datetime is parsed correctly
            if 'Datetime' in df.columns:
                df['Datetime'] = pd.to_datetime(df['Datetime'])
            elif 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            logger.error(f"Error loading cached data for {ticker_symbol}: {e}. Fetching new data.")
    
    logger.info(f"Fetching data for {ticker_symbol} from yfinance...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {ticker_symbol}")
            return None
            
        # Reset index to make Datetime a column
        df.reset_index(inplace=True)
        
        # Ensure timezone is removed or handled consistently
        if 'Datetime' in df.columns:
            df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved data to {filepath}")
            
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {ticker_symbol}: {e}")
        return None

def get_market_cap(ticker_symbol):
    """
    Get market cap for a stock in Crores INR.
    
    Args:
        ticker_symbol (str): Stock ticker symbol.
        
    Returns:
        float: Market cap in Crores, or None if not found.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        market_cap = info.get('marketCap')
        
        if market_cap:
            # yfinance returns market cap in absolute currency (usually INR for .NS)
            # Convert to Crores (1 Crore = 10,000,000)
            return market_cap / 10000000
        return None
    except Exception as e:
        logger.error(f"Error fetching market cap for {ticker_symbol}: {e}")
        return None

def get_earnings_dates(ticker_symbol):
    """
    Get earnings dates for a stock.
    
    Args:
        ticker_symbol (str): Stock ticker symbol.
        
    Returns:
        pd.DataFrame: Earnings dates.
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.earnings_dates
    except Exception as e:
        logger.error(f"Error fetching earnings dates for {ticker_symbol}: {e}")
        return None

def save_raw_data(df, ticker_symbol, filename=None):
    """
    Save raw data to CSV.
    """
    if df is None:
        return
        
    if filename is None:
        filename = f"{ticker_symbol}_raw.csv"
        
    filepath = os.path.join(RAW_DIR, filename)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved data to {filepath}")

def load_raw_data(ticker_symbol):
    """
    Load raw data from CSV.
    """
    filename = f"{ticker_symbol}_raw.csv"
    filepath = os.path.join(RAW_DIR, filename)
    
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    else:
        logger.warning(f"File not found: {filepath}")
        return None

if __name__ == "__main__":
    # Test with a sample stock
    symbol = "RELIANCE.NS"
    df = fetch_stock_data(symbol)
    if df is not None:
        print(df.head())
        save_raw_data(df, symbol)
        
    mcap = get_market_cap(symbol)
    print(f"Market Cap: {mcap} Cr")
    
    earnings = get_earnings_dates(symbol)
    if earnings is not None:
        print(earnings.head())
