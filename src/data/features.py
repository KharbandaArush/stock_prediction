import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def add_technical_indicators(df):
    """
    Add technical indicators to the DataFrame.
    """
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['Close', 'High', 'Low', 'Volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns for technical indicators: {required_cols}")
        return df

    # RSI (14 periods)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands (20 periods, 2 std dev)
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['STD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['STD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['STD20'] * 2)
    
    # ATR (14 periods)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    return df

def add_volume_features(df):
    """
    Add volume-related features, including Volume in INR.
    """
    df = df.copy()
    
    # Volume in INR (Price * Volume)
    # Using Close price as approximation for the interval
    df['Volume_INR'] = df['Close'] * df['Volume']
    
    # Volume Moving Averages
    df['Vol_MA20'] = df['Volume'].rolling(window=20).mean()
    df['Vol_Ratio'] = df['Volume'] / df['Vol_MA20']
    
    # VWAP (Volume Weighted Average Price)
    # For minute data, we can calculate a rolling VWAP or daily VWAP
    # Here we calculate a rolling VWAP for the last 60 periods (approx 1 hour)
    v = df['Volume'].values
    p = df['Close'].values
    df['VWAP'] = (df['Volume'] * df['Close']).rolling(window=60).sum() / df['Volume'].rolling(window=60).sum()
    
    return df

def add_time_features(df):
    """
    Add time-based features.
    """
    df = df.copy()
    
    if 'Datetime' in df.columns:
        df['Hour'] = df['Datetime'].dt.hour
        df['Minute'] = df['Datetime'].dt.minute
        df['DayOfWeek'] = df['Datetime'].dt.dayofweek
    
    # Lagged features (e.g., returns)
    df['Return_1m'] = df['Close'].pct_change()
    df['Return_5m'] = df['Close'].pct_change(periods=5)
    df['Return_15m'] = df['Close'].pct_change(periods=15)
    
    # Volatility (rolling std dev of returns)
    df['Volatility_15m'] = df['Return_1m'].rolling(window=15).std()
    
    return df

def add_result_features(df, earnings_dates):
    """
    Add features related to result announcements.
    
    Args:
        df (pd.DataFrame): Price data with Datetime index or column.
        earnings_dates (pd.DataFrame): Earnings dates from yfinance.
    """
    df = df.copy()
    
    if earnings_dates is None or earnings_dates.empty:
        df['Days_Since_Result'] = -1
        df['Days_To_Result'] = -1
        return df
        
    # Process earnings dates
    # yfinance earnings_dates index is Timestamp
    earnings_dates = earnings_dates.index.sort_values()
    
    # Ensure df has Datetime column
    if 'Datetime' not in df.columns:
         # If index is DatetimeIndex, reset it
         if isinstance(df.index, pd.DatetimeIndex):
             df = df.reset_index()
         else:
             logger.warning("No Datetime column found for result features")
             return df

    # Function to find nearest earnings dates
    def get_days_since(current_date, dates):
        past_dates = dates[dates < current_date]
        if past_dates.empty:
            return -1
        return (current_date - past_dates[-1]).days

    def get_days_to(current_date, dates):
        future_dates = dates[dates > current_date]
        if future_dates.empty:
            return -1
        return (future_dates[0] - current_date).days

    # This can be slow for large dataframes, applying row-wise
    # For efficiency, we can use merge_asof if needed, but for now apply is fine for 7 days data
    # Note: earnings dates are usually just dates, while current_date has time
    # Normalize to date for comparison
    
    # Optimization: Pre-calculate for unique dates in df
    unique_dates = df['Datetime'].dt.date.unique()
    date_map_since = {}
    date_map_to = {}
    
    earnings_dates_date = earnings_dates.date
    
    for d in unique_dates:
        # Convert d to timestamp for comparison if needed, or compare dates
        # Simple loop over sorted earnings dates
        
        # Since
        past = [ed for ed in earnings_dates_date if ed < d]
        if past:
            date_map_since[d] = (d - past[-1]).days
        else:
            date_map_since[d] = -1
            
        # To
        future = [ed for ed in earnings_dates_date if ed > d]
        if future:
            date_map_to[d] = (future[0] - d).days
        else:
            date_map_to[d] = -1
            
    df['Days_Since_Result'] = df['Datetime'].dt.date.map(date_map_since)
    df['Days_To_Result'] = df['Datetime'].dt.date.map(date_map_to)
    
    return df

def generate_features(df, earnings_dates=None):
    """
    Master function to generate all features.
    """
    df = add_technical_indicators(df)
    df = add_volume_features(df)
    df = add_time_features(df)
    if earnings_dates is not None:
        df = add_result_features(df, earnings_dates)
    else:
        df['Days_Since_Result'] = -1
        df['Days_To_Result'] = -1
        
    # Drop NaN values created by rolling windows
    df.dropna(inplace=True)
    
    return df
