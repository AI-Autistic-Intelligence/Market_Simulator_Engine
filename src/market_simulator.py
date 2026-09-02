import os
import json
import numpy as np
import pandas as pd

def generate_stochastic_market(seed, num_candles=2500):
    np.random.seed(seed)
    
    # Base parameters
    S0 = 100.0  # Initial price
    mu = 0.0001  # Drift
    base_sigma = 0.01  # Base volatility
    
    prices = [S0]
    highs = []
    lows = []
    vols = []
    
    # Stochastic process with Volatility Clustering (GARCH proxy) and Jumps
    current_sigma = base_sigma
    
    for i in range(num_candles):
        # Volatility clustering (autoregressive volatility)
        current_sigma = max(0.001, current_sigma + np.random.normal(0, 0.002))
        if current_sigma > 0.05: current_sigma *= 0.9  # Mean reversion of vol
        
        # Jump diffusion (Merton)
        jump = 0
        if np.random.rand() < 0.05:  # 5% chance of a jump
            jump = np.random.normal(0, 0.05)
            
        # GBM step (Simulate 10 intra-candle ticks)
        ticks = [prices[-1]]
        for _ in range(10):
            dW = np.random.normal(0, np.sqrt(1/10))
            dS = mu * ticks[-1] * (1/10) + current_sigma * ticks[-1] * dW
            ticks.append(ticks[-1] + dS)
            
        open_p = ticks[0]
        close_p = ticks[-1] + (jump * ticks[-1])
        high_p = max(ticks)
        low_p = min(ticks)
        
        # Ensure High/Low consistency with Open/Close
        high_p = max(high_p, open_p, close_p)
        low_p = min(low_p, open_p, close_p)
        
        # Volume correlates with volatility and jump size
        vol = int(np.random.lognormal(mean=10, sigma=1) * (1 + current_sigma*10 + abs(jump)*10))
        
        prices.append(close_p)
        highs.append(high_p)
        lows.append(low_p)
        vols.append(vol)
        
    # Build dataframe
    df = pd.DataFrame({
        'open': prices[:-1],
        'high': highs,
        'low': lows,
        'close': prices[1:],
        'volume': vols,
        'spread': np.random.uniform(0.01, 0.05, num_candles)
    })
    
    # Add dummy timestamp
    df['timestamp'] = [int(1600000000000 + (i * 60000)) for i in range(num_candles)]
    
    return df

def generate_all_markets(num_markets=1000, output_dir="../data/markets", batch_offset=0):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_markets} stochastic markets (Batch Offset: {batch_offset})...")
    
    for i in range(1, num_markets + 1):
        # Unique seed per market per batch to ensure entirely new data
        seed = i + 1000 + (batch_offset * 10000)
        df = generate_stochastic_market(seed=seed)
        
        # Format required by downstream processes (array of dicts)
        candles = df.to_dict('records')
        
        filename = f"market_{i:04d}.json"
        with open(os.path.join(output_dir, filename), 'w') as f:
            json.dump(candles, f, indent=2)
            
        if i % 100 == 0:
            print(f"[{i}/{num_markets}] Generated {filename}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()
    
    generate_all_markets(num_markets=args.markets, batch_offset=args.batch)
