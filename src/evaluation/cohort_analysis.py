import pandas as pd
import numpy as np
from src.evaluation.metrics import evaluate_model

def assign_cohorts(df, market_cap_map):
    """
    Assign Market Cap and Volume cohorts.
    
    Args:
        df: DataFrame with 'Ticker' and 'Volume_INR'.
        market_cap_map: Dict mapping Ticker -> Market Cap (Cr).
    """
    df = df.copy()
    
    # Market Cap Cohort
    def get_mcap_cohort(ticker):
        mcap = market_cap_map.get(ticker)
        if mcap is None: return 'Unknown'
        if mcap < 200: return '<200cr'
        if mcap < 500: return '200-500cr'
        if mcap < 1000: return '500-1000cr'
        if mcap < 2000: return '1000-2000cr'
        return '>2000cr'
        
    df['MarketCap_Cohort'] = df['Ticker'].apply(get_mcap_cohort)
    
    # Volume Cohort (Daily Volume INR)
    # We calculate deciles or fixed buckets
    # Here using simple quantiles based on the provided data
    if 'Volume_INR' in df.columns:
        df['Volume_Cohort'] = pd.qcut(df['Volume_INR'], 3, labels=['Low Vol', 'Med Vol', 'High Vol'])
    else:
        df['Volume_Cohort'] = 'Unknown'
        
    return df

def analyze_cohorts(df_results):
    """
    Analyze performance by cohort.
    """
    # Group by Market Cap Cohort
    mcap_performance = df_results.groupby('MarketCap_Cohort').apply(lambda x: pd.Series(evaluate_model(x)))
    
    # Group by Volume Cohort
    vol_performance = df_results.groupby('Volume_Cohort').apply(lambda x: pd.Series(evaluate_model(x)))
    
    return mcap_performance, vol_performance
