import os
import json
import yfinance as yf
import pandas as pd
import numpy as np

def fetch_and_save_real_markets(output_dir="../data/markets"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Forex, Crypto, Indices, Commodities
    tickers = {
        "EURUSD=X": "EURUSD",
        "GBPUSD=X": "GBPUSD",
        "JPY=X": "USDJPY",
        "AUDUSD=X": "AUDUSD",
        "BTC-USD": "BTCUSD",
        "ETH-USD": "ETHUSD",
        "^GSPC": "SP500",
        "^IXIC": "NASDAQ",
        "^DJI": "DOW",
        "GC=F": "GOLD",
        "CL=F": "OIL"
    }
    
    print("Fetching real market data from Yahoo Finance...")
    
    market_index = 2000 # Save real markets with a high index to avoid clashing with 0001
    
    for symbol, name in tickers.items():
        try:
            print(f"Downloading {name} ({symbol}) - 1h timeframe, max available...")
            # We get max data for 1h (yfinance usually allows 730 days for 1h)
            # Actually, yfinance 1h is max 730 days
            df = yf.download(symbol, interval="1h", period="730d", progress=False)
            
            if df.empty:
                print(f"❌ Failed to download {name}")
                continue
                
            # Formatting to match our market_simulator.py format
            # Required columns: timestamp, open, high, low, close, volume, spread
            # Note: yf returns multi-index columns for some versions, let's flatten if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            df = df.reset_index()
            
            # yfinance returns Datetime, Open, High, Low, Close, Adj Close, Volume
            formatted = pd.DataFrame()
            
            # convert datetime to timestamp ms
            time_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
            if time_col in df.columns:
                formatted['timestamp'] = df[time_col].astype(np.int64) // 10**6
            else:
                formatted['timestamp'] = [int(1600000000000 + (i * 3600000)) for i in range(len(df))]
                
            formatted['open'] = df['Open']
            formatted['high'] = df['High']
            formatted['low'] = df['Low']
            formatted['close'] = df['Close']
            
            # Volume can be 0 for Forex. We can inject a fake volume based on volatility if 0
            vols = df['Volume'].values
            if vols.sum() == 0:
                # Approximate volume using candle size for forex
                vols = (abs(df['High'] - df['Low']) / df['Close']) * 10000000
                vols = vols.astype(int)
            formatted['volume'] = vols
            
            # Approximate spread as 0.01% of price
            formatted['spread'] = df['Close'] * 0.0001 
            
            # Drop NAs
            formatted = formatted.dropna()
            
            candles = formatted.to_dict('records')
            
            filename = f"market_{market_index:04d}_REAL_{name}.json"
            with open(os.path.join(output_dir, filename), 'w') as f:
                json.dump(candles, f, indent=2)
                
            print(f"✅ Saved {filename} ({len(candles)} candles)")
            market_index += 1
            
        except Exception as e:
            print(f"❌ Error with {name}: {e}")

if __name__ == "__main__":
    fetch_and_save_real_markets()
