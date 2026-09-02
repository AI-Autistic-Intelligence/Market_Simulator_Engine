import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def generate_random_walk(num_candles=5000, start_price=100.0, volatility=0.001, drift=0.0):
    prices = [start_price]
    for _ in range(num_candles):
        # Geometric Brownian Motion simulation
        change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    df = pd.DataFrame({'Close': prices[1:]})
    
    # Generate O, H, L relative to Close and previous Close
    prev_close = np.roll(df['Close'].values, shift=1)
    prev_close[0] = start_price
    
    # Open is generally near the prev close with some gap noise
    df['Open'] = prev_close * (1 + np.random.normal(0, volatility * 0.1, num_candles))
    
    # High must be max(Open, Close) + some positive noise
    df['High'] = np.maximum(df['Open'], df['Close']) * (1 + np.abs(np.random.normal(0, volatility * 0.5, num_candles)))
    
    # Low must be min(Open, Close) - some positive noise
    df['Low'] = np.minimum(df['Open'], df['Close']) * (1 - np.abs(np.random.normal(0, volatility * 0.5, num_candles)))
    
    # Volume is randomized
    df['Volume'] = np.random.randint(100, 10000, num_candles)
    
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def generate_1000_markets():
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating 1000 Synthetic Markets...")
    
    for i in tqdm(range(1, 1001)):
        # Randomize market conditions
        volatility = np.random.uniform(0.0005, 0.003) # low to high volatility
        drift = np.random.uniform(-0.0001, 0.0001)    # slight up/down trends
        start_price = np.random.uniform(10.0, 500.0)
        
        df = generate_random_walk(num_candles=5000, start_price=start_price, volatility=volatility, drift=drift)
        
        filename = os.path.join(output_dir, f"market_{i:04d}.csv")
        df.to_csv(filename, index=False)

if __name__ == "__main__":
    generate_1000_markets()
    print("Done! 1000 markets generated.")
