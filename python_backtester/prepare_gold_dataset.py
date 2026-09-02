import pandas as pd
import numpy as np
import os
import sys
from tqdm import tqdm

WINDOW_SIZE = 30
HOLD_MAX_CANDLES = 60

# Constants matching TSO_AI.mq5
ATR_STOP_MULT = 1.5
ATR_TAKE_MULT = 3.0
TRAILING_START_MULT = 1.0
TRAILING_STEP_MULT = 0.5
# We don't use hardcoded spread anymore; we read from CSV
SLIPPAGE_POINTS = 30  # Penalty for 38ms ping and market execution slippage

def compute_rsi(closes, window=14):
    deltas = np.diff(closes)
    seed = deltas[:window]
    up = seed[seed >= 0].sum()/window
    down = -seed[seed < 0].sum()/window
    rs = up/down if down != 0 else 0
    rsi = np.zeros_like(closes)
    rsi[:window] = 50
    if down == 0:
        rsi[window] = 100
    else:
        rsi[window] = 100. - 100./(1.+rs)

    for i in range(window, len(closes)-1):
        delta = deltas[i]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(window-1) + upval)/window
        down = (down*(window-1) + downval)/window

        rs = up/down if down != 0 else 0
        rsi[i+1] = 100. - 100./(1.+rs) if down != 0 else 100.
    return rsi

def compute_ema(closes, window):
    alpha = 2.0 / (window + 1.0)
    ema = np.zeros_like(closes)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = closes[i] * alpha + ema[i-1] * (1.0 - alpha)
    return ema

def simulate_trade(idx, opens, highs, lows, closes, spreads_points, atr_val, point_val, is_long):
    # spreads_points is the spread in MT5 points. E.g. 37. point_val = 0.01 or 0.001
    spread_price = spreads_points[idx] * point_val
    slippage_penalty = SLIPPAGE_POINTS * point_val
    entry_price = closes[idx] + (spread_price if is_long else -spread_price)
    
    current_sl = entry_price - (atr_val * ATR_STOP_MULT) if is_long else entry_price + (atr_val * ATR_STOP_MULT)
    current_tp = entry_price + (atr_val * ATR_TAKE_MULT) if is_long else entry_price - (atr_val * ATR_TAKE_MULT)
    trailing_start = atr_val * TRAILING_START_MULT
    trailing_step = atr_val * TRAILING_STEP_MULT
    
    for t in range(idx+1, min(idx + HOLD_MAX_CANDLES, len(closes))):
        low = lows[t]
        high = highs[t]
        close = closes[t]
        
        if is_long:
            if close - entry_price > trailing_start:
                new_sl = close - trailing_step
                if new_sl > current_sl and new_sl > entry_price:
                    current_sl = new_sl
                    
            if low <= current_sl:
                return (current_sl - entry_price) - slippage_penalty
            if high >= current_tp:
                return (current_tp - entry_price) - slippage_penalty
        else:
            if entry_price - close > trailing_start:
                new_sl = close + trailing_step
                if new_sl < current_sl or current_sl == 0:
                    if new_sl < entry_price:
                        current_sl = new_sl
                        
            if high >= current_sl:
                return (entry_price - current_sl) - slippage_penalty
            if low <= current_tp:
                return (entry_price - current_tp) - slippage_penalty
                
    # Max hold time reached, close at market
    final_close = closes[min(idx + HOLD_MAX_CANDLES - 1, len(closes) - 1)]
    raw_pnl = (final_close - entry_price) if is_long else (entry_price - final_close)
    return raw_pnl - slippage_penalty

def main():
    print("Loading GOLD dataset...")
    df = pd.read_csv("../data/markets/GOLD_M1.csv")
    
    closes = df['Close'].values.flatten()
    opens = df['Open'].values.flatten()
    highs = df['High'].values.flatten()
    lows = df['Low'].values.flatten()
    vols = df['TickVolume'].values.flatten()
    spreads = df['Spread'].values.flatten()
    
    print("Calculating indicators...")
    rsi = compute_rsi(closes, 14)
    ema9 = compute_ema(closes, 9)
    ema21 = compute_ema(closes, 21)
    
    df_sma20 = pd.Series(closes).rolling(window=20).mean().values
    df_std20 = pd.Series(closes).rolling(window=20).std(ddof=0).values
    bb_up = df_sma20 + 2 * df_std20
    bb_dn = df_sma20 - 2 * df_std20
    
    high_low = highs - lows
    high_close = np.abs(highs - np.roll(closes, 1))
    low_close = np.abs(lows - np.roll(closes, 1))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = pd.Series(tr).rolling(14).mean().values
    
    all_X = []
    all_Y_pnl = []
    total_theoretical_profit = 0.0
    total_perfect_trades = 0
    
    print("Generating Training Data from GOLD (50,000 candles)...")
    
    # Calculate point value automatically
    point_val = 0.01 if closes[0] < 10000 else 0.001
    if closes[0] < 100:
        point_val = 0.0001
    print(f"Detected point value: {point_val}")
    
    for i in tqdm(range(25, len(df) - WINDOW_SIZE - HOLD_MAX_CANDLES)):
        window_lows = lows[i:i+WINDOW_SIZE]
        window_highs = highs[i:i+WINDOW_SIZE]
        window_vols = vols[i:i+WINDOW_SIZE]
        
        min_p = np.min(window_lows)
        max_p = np.max(window_highs)
        max_v = np.max(window_vols)
        
        price_range = max_p - min_p
        if price_range <= 0.0:
            price_range = 1e-9
            
        if np.isnan(bb_up[i]) or np.isnan(atr[i+WINDOW_SIZE-1]):
            continue
            
        x_window = []
        for t in range(WINDOW_SIZE):
            idx = i + t
            c_o = (opens[idx] - min_p) / price_range
            c_h = (highs[idx] - min_p) / price_range
            c_l = (lows[idx] - min_p) / price_range
            c_c = (closes[idx] - min_p) / price_range
            c_v = vols[idx] / max_v if max_v > 0 else 0.0
            
            c_rsi = rsi[idx] / 100.0
            c_ema9 = (ema9[idx] - min_p) / price_range
            c_ema21 = (ema21[idx] - min_p) / price_range
            c_bbup = (bb_up[idx] - min_p) / price_range
            c_bbdn = (bb_dn[idx] - min_p) / price_range
            c_spread = (spreads[idx] * point_val) / price_range
            
            x_window.append([c_o, c_h, c_l, c_c, c_v, c_rsi, c_ema9, c_ema21, c_bbup, c_bbdn, c_spread])
        
        target_idx = i + WINDOW_SIZE - 1
        atr_val = atr[target_idx]
        if np.isnan(atr_val) or atr_val == 0:
            continue
            
        pnl_long = simulate_trade(target_idx, opens, highs, lows, closes, spreads, atr_val, point_val, True)
        pnl_short = simulate_trade(target_idx, opens, highs, lows, closes, spreads, atr_val, point_val, False)
        
        best_pnl = max(pnl_long, pnl_short)
        if best_pnl > 0:
            total_theoretical_profit += best_pnl
            total_perfect_trades += 1
        
        all_X.append(x_window)
        # Normalize PnL slightly by ATR
        all_Y_pnl.append([pnl_long / atr_val, pnl_short / atr_val])
        
    all_X = np.array(all_X, dtype=np.float32)
    all_Y_pnl = np.array(all_Y_pnl, dtype=np.float32)
    
    print(f"\nGOLD Dataset Extraction Complete: {len(all_X)} samples")
    print(f"Total Theoretical Max Profit from GOLD: +{total_theoretical_profit:,.2f} points")
    print(f"Total Perfect Trades found: {total_perfect_trades}")
    
    os.makedirs("../data/cache_gold", exist_ok=True)
    np.save("../data/cache_gold/X_features.npy", all_X)
    np.save("../data/cache_gold/Y_pnl.npy", all_Y_pnl)
    print("Saved to ../data/cache_gold/")

if __name__ == "__main__":
    main()
