import pandas as pd
import yfinance as yf
import requests
import io
import logging
import os
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')

def fetch_all_nse_tickers():
    """
    Fetch list of all NSE equity tickers.
    """
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36'
    }
    
    try:
        logger.info(f"Downloading ticker list from {url}...")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
        
        tickers = df['SYMBOL'].tolist()
        tickers = [f"{t}.NS" for t in tickers]
        logger.info(f"Found {len(tickers)} tickers.")
        return tickers
    except Exception as e:
        logger.error(f"Failed to fetch from NSE: {e}")
        return []

def filter_by_market_cap(tickers, min_cr=200, max_cr=2000, output_file=None):
    """
    Filter tickers by market cap (in Crores) and save incrementally.
    """
    filtered_tickers = []
    batch_size = 50
    total = len(tickers)
    
    logger.info(f"Filtering {total} tickers by Market Cap ({min_cr}-{max_cr} Cr)...")
    
    # Initialize output file with headers if provided and file doesn't exist
    if output_file and not os.path.exists(output_file):
        pd.DataFrame(columns=['Ticker', 'MarketCap_Cr']).to_csv(output_file, index=False)

    start_index = 150 # Resume from 150
    for i in range(start_index, total, batch_size):
        batch = tickers[i:i+batch_size]
        batch_results = []
        
        for ticker_symbol in batch:
            try:
                # Standard Ticker call, let yfinance handle session
                dat = yf.Ticker(ticker_symbol)
                
                # Use info as fast_info proved unreliable
                mcap = None
                try:
                    info = dat.info
                    mcap = info.get('marketCap')
                except Exception:
                    pass
                
                if mcap:
                    mcap_cr = mcap / 10000000
                    # logger.info(f"{ticker_symbol}: {mcap_cr:.2f} Cr")
                    if min_cr <= mcap_cr <= max_cr:
                        result = {
                            'Ticker': ticker_symbol,
                            'MarketCap_Cr': mcap_cr
                        }
                        filtered_tickers.append(result)
                        batch_results.append(result)
                else:
                    # logger.warning(f"No market cap for {ticker_symbol}")
                    pass
                
                # Be polite
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error checking {ticker_symbol}: {e}")
                pass
        
        # Save batch results incrementally
        if output_file and batch_results:
            pd.DataFrame(batch_results).to_csv(output_file, mode='a', header=False, index=False)
            logger.info(f"Saved {len(batch_results)} new matches to {output_file}")

        logger.info(f"Processed {min(i+batch_size, total)}/{total}. Found {len(filtered_tickers)} matches so far.")
        
    return pd.DataFrame(filtered_tickers)

if __name__ == "__main__":
    tickers = fetch_all_nse_tickers()
    if tickers:
        os.makedirs(DATA_DIR, exist_ok=True)
        output_path = os.path.join(DATA_DIR, 'filtered_tickers.csv')
        
        # Process all tickers and save incrementally
        df_filtered = filter_by_market_cap(tickers, min_cr=200, max_cr=2000, output_file=output_path)
        
        print("\nFiltered Stocks:")
        print(df_filtered.head())
        logger.info(f"Total {len(df_filtered)} tickers saved to {output_path}")
