import numpy as np
import pandas as pd

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def calculate_ema(prices, window=9):
    return prices.ewm(span=window, adjust=False).mean()

def calculate_bollinger_bands(prices, window=20, num_std=2):
    sma = prices.rolling(window=window, min_periods=1).mean()
    std = prices.rolling(window=window, min_periods=1).std()
    return sma + (std * num_std), sma - (std * num_std)

def generate_dataset(candles_df, window_size=30):
    """
    Returns X (features) and Y (target prob) identically to TS version.
    Features: 11 (open, high, low, close, volume, rsi, ema9, ema21, bb_upper, bb_lower, spread)
    """
    df = candles_df.copy()
    
    # Calculate indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['ema9'] = calculate_ema(df['close'], 9)
    df['ema21'] = calculate_ema(df['close'], 21)
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'])
    
    # Fill NAs
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    
    dataset_x = []
    dataset_y_prob = []
    dataset_y_ohlc = []
    
    for i in range(len(df) - window_size):
        window = df.iloc[i:i + window_size]
        target = df.iloc[i + window_size]
        
        # Min/Max scaling purely on window values
        min_price = window['low'].min()
        max_price = window['high'].max()
        max_vol = window['volume'].max()
        
        price_range = max_price - min_price if max_price > min_price else 1e-9
        
        def norm(val): return (val - min_price) / price_range
        def norm_vol(val): return val / max_vol if max_vol > 0 else 0
        
        x_features = []
        for _, row in window.iterrows():
            x_features.append([
                norm(row['open']),
                norm(row['high']),
                norm(row['low']),
                norm(row['close']),
                norm_vol(row['volume']),
                row['rsi'] / 100.0,
                norm(row['ema9']),
                norm(row['ema21']),
                norm(row['bb_upper']),
                norm(row['bb_lower']),
                row['spread'] / price_range if price_range > 0 else 0
            ])
            
        is_up = 1 if target['close'] >= window.iloc[-1]['close'] else 0
        
        dataset_x.append(x_features)
        dataset_y_prob.append(is_up)
        dataset_y_ohlc.append([
            norm(target['open']), norm(target['high']), 
            norm(target['low']), norm(target['close'])
        ])
        
    return np.array(dataset_x, dtype=np.float32), np.array(dataset_y_prob, dtype=np.float32), np.array(dataset_y_ohlc, dtype=np.float32)
