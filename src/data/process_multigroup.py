import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data')
INPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'multigroup.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'filtered_tickers.csv')

def process_multigroup_data():
    """
    Process multigroup.csv to generate filtered_tickers.csv.
    """
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file not found: {INPUT_FILE}")
        return

    try:
        logger.info(f"Reading input file: {INPUT_FILE}")
        df = pd.read_csv(INPUT_FILE)
        
        # Filter rows where NSE Code is present
        df_filtered = df[df['NSE Code'].notna() & (df['NSE Code'] != '')].copy()
        
        logger.info(f"Found {len(df_filtered)} rows with valid NSE Code.")
        
        # Format Ticker with .NS suffix
        df_filtered['Ticker'] = df_filtered['NSE Code'].apply(lambda x: f"{str(x).strip()}.NS")
        
        # Rename Market Capitalization to MarketCap_Cr
        # Assuming Market Capitalization is already in Crores based on typical Indian data sources like Trendlyne/Screener
        # If it's not, we might need conversion, but usually these exports are in Cr.
        # Let's check the file content preview again... 
        # Line 2: 20 Microns Ltd. ... Market Capitalization: 675.17
        # Line 100: Apex Frozen Foods ... Market Capitalization: 928.97
        # These look like Crores.
        
        df_filtered['MarketCap_Cr'] = df_filtered['Market Capitalization']
        
        # Select required columns
        final_df = df_filtered[['Ticker', 'MarketCap_Cr']]
        
        # Save to CSV
        os.makedirs(DATA_DIR, exist_ok=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        logger.info(f"Saved {len(final_df)} tickers to {OUTPUT_FILE}")
        
        print("\nProcessed Tickers:")
        print(final_df.head())

    except Exception as e:
        logger.error(f"Error processing data: {e}")

if __name__ == "__main__":
    process_multigroup_data()
