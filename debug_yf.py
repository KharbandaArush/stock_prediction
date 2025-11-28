import yfinance as yf

def test_yf():
    ticker = "RELIANCE.NS"
    print(f"Testing {ticker}...")
    
    try:
        dat = yf.Ticker(ticker)
        
        print("Attempting fast_info...")
        mcap = dat.fast_info.get('market_cap')
        print(f"Fast Info Market Cap: {mcap}")
        
        print("Attempting info...")
        info = dat.info
        mcap_info = info.get('marketCap')
        print(f"Info Market Cap: {mcap_info}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_yf()
