import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from ai_inference import parse_mqh_weights, build_model, calculate_indicators
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prepare_features(df):
    WINDOW_SIZE = 30
    SPREAD = 0.0001
    num_samples = len(df) - WINDOW_SIZE
    features = np.zeros((num_samples, WINDOW_SIZE, 11), dtype=np.float32)
    
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    vols = df['Volume'].values
    rsis = df['RSI'].values
    ema9s = df['EMA9'].values
    ema21s = df['EMA21'].values
    bbups = df['BB_UP'].values
    bbdns = df['BB_DN'].values
    
    for i in range(num_samples):
        start_idx = i
        end_idx = i + WINDOW_SIZE
        
        min_price = np.min([opens[start_idx:end_idx], highs[start_idx:end_idx], lows[start_idx:end_idx], closes[start_idx:end_idx]])
        max_price = np.max([opens[start_idx:end_idx], highs[start_idx:end_idx], lows[start_idx:end_idx], closes[start_idx:end_idx]])
        price_range = max_price - min_price
        if price_range == 0: price_range = 1.0
        
        features[i, :, 0] = (opens[start_idx:end_idx] - min_price) / price_range
        features[i, :, 1] = (highs[start_idx:end_idx] - min_price) / price_range
        features[i, :, 2] = (lows[start_idx:end_idx] - min_price) / price_range
        features[i, :, 3] = (closes[start_idx:end_idx] - min_price) / price_range
        features[i, :, 4] = vols[start_idx:end_idx] / 10000.0
        features[i, :, 5] = rsis[start_idx:end_idx] / 100.0
        features[i, :, 6] = (ema9s[start_idx:end_idx] - min_price) / price_range
        features[i, :, 7] = (ema21s[start_idx:end_idx] - min_price) / price_range
        features[i, :, 8] = (bbups[start_idx:end_idx] - min_price) / price_range
        features[i, :, 9] = (bbdns[start_idx:end_idx] - min_price) / price_range
        features[i, :, 10] = SPREAD / 100.0
        
    return features

def backtest_market(df, model):
    WINDOW_SIZE = 30
    CONFIDENCE_THRESHOLD = 0.55
    ATR_STOP_MULT = 1.5
    ATR_TAKE_MULT = 3.0
    TRAILING_START_MULT = 1.0
    TRAILING_STEP_MULT = 0.5
    SPREAD = 0.0001
    
    features = prepare_features(df)
    action_preds = model.predict(features, batch_size=256, verbose=0)
    
    position = None 
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    
    trades = []
    
    for i in range(WINDOW_SIZE, len(df)):
        current_close = df.loc[i, 'Close']
        current_high = df.loc[i, 'High']
        current_low = df.loc[i, 'Low']
        current_atr = df.loc[i, 'ATR']
        
        if position == 'LONG':
            if current_close - entry_price > current_atr * TRAILING_START_MULT:
                new_sl = current_close - (current_atr * TRAILING_STEP_MULT)
                if new_sl > sl and new_sl > entry_price:
                    sl = new_sl
            
            if current_low <= sl:
                trades.append({'type': 'LONG', 'pnl': sl - entry_price})
                position = None
            elif current_high >= tp:
                trades.append({'type': 'LONG', 'pnl': tp - entry_price})
                position = None
                
        elif position == 'SHORT':
            if entry_price - current_close > current_atr * TRAILING_START_MULT:
                new_sl = current_close + (current_atr * TRAILING_STEP_MULT)
                if new_sl < sl or sl == 0:
                    if new_sl < entry_price:
                        sl = new_sl
            
            if current_high >= sl:
                trades.append({'type': 'SHORT', 'pnl': entry_price - sl})
                position = None
            elif current_low <= tp:
                trades.append({'type': 'SHORT', 'pnl': entry_price - tp})
                position = None
                
        if position is None:
            prob_long = action_preds[i - WINDOW_SIZE][0]
            prob_short = action_preds[i - WINDOW_SIZE][1]
            
            if prob_long >= CONFIDENCE_THRESHOLD:
                position = 'LONG'
                entry_price = current_close + SPREAD
                sl = entry_price - (current_atr * ATR_STOP_MULT)
                tp = entry_price + (current_atr * ATR_TAKE_MULT)
            elif prob_short >= CONFIDENCE_THRESHOLD:
                position = 'SHORT'
                entry_price = current_close - SPREAD
                sl = entry_price + (current_atr * ATR_STOP_MULT)
                tp = entry_price - (current_atr * ATR_TAKE_MULT)
                
    if position == 'LONG':
        trades.append({'type': 'LONG', 'pnl': df.iloc[-1]['Close'] - entry_price})
    elif position == 'SHORT':
        trades.append({'type': 'SHORT', 'pnl': entry_price - df.iloc[-1]['Close']})
        
    return trades

def main():
    print("Loading AI model from CATWeights.mqh...")
    mqh_path = os.path.join(os.path.dirname(__file__), '..', 'mql5_bot', 'CATWeights.mqh')
    if not os.path.exists(mqh_path):
        print("ERROR: CATWeights.mqh non ancora generato. Attendi la fine dell'addestramento.")
        return
        
    weights = parse_mqh_weights(mqh_path)
    model = build_model(weights)
    print("AI Model loaded successfully.")
    
    files = glob.glob(os.path.join(os.path.dirname(__file__), '..', 'data', 'markets', '*.csv'))
    print(f"Found {len(files)} synthetic markets to backtest. Starting evaluation...")
    
    market_reports = []
    
    global_trades = 0
    global_wins = 0
    global_pnl = 0.0
    
    for file in tqdm(files):
        df = pd.read_csv(file)
        df = calculate_indicators(df)
        
        trades = backtest_market(df, model)
        
        market_name = os.path.basename(file).replace('.csv', '')
        t_trades = len(trades)
        t_wins = sum(1 for t in trades if t['pnl'] > 0)
        t_pnl = sum(t['pnl'] for t in trades)
        t_winrate = (t_wins / t_trades * 100) if t_trades > 0 else 0.0
        
        # Calculate Peak and Drawdown
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for t in trades:
            equity += t['pnl']
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd
        
        market_reports.append({
            'Market': market_name,
            'Trades': t_trades,
            'Wins': t_wins,
            'WinRate(%)': round(t_winrate, 2),
            'NetProfit': round(t_pnl, 5),
            'MaxDrawdown': round(max_dd, 5)
        })
        
        global_trades += t_trades
        global_wins += t_wins
        global_pnl += t_pnl
                
    global_winrate = (global_wins / global_trades) * 100 if global_trades > 0 else 0
    
    # Save Market-by-Market Report
    report_df = pd.DataFrame(market_reports)
    report_df.to_csv("backtest_detailed_report.csv", index=False)
    
    # Generate Text Report
    with open("backtest_summary.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write("🏆 RISULTATI BACKTEST 1000 MERCATI SINTETICI 🏆\n")
        f.write("="*50 + "\n")
        f.write(f"Totale Trades Effettuati : {global_trades}\n")
        f.write(f"Totale Vittorie          : {global_wins}\n")
        f.write(f"Win Rate Finale          : {global_winrate:.2f}%\n")
        f.write(f"Profitto Netto Globale   : {global_pnl:.5f} punti\n")
        f.write("="*50 + "\n")
        f.write("Mercati Migliori (Top 5 per Profitto):\n")
        top_5 = report_df.sort_values(by="NetProfit", ascending=False).head(5)
        for _, row in top_5.iterrows():
            f.write(f" - {row['Market']}: Profitto {row['NetProfit']} | WR: {row['WinRate(%)']}%\n")
        
        f.write("\nMercati Peggiori (Flop 5 per Profitto):\n")
        flop_5 = report_df.sort_values(by="NetProfit", ascending=True).head(5)
        for _, row in flop_5.iterrows():
            f.write(f" - {row['Market']}: Profitto {row['NetProfit']} | WR: {row['WinRate(%)']}%\n")
            
        f.write("\nStatistiche dettagliate mercato per mercato esportate in 'backtest_detailed_report.csv'.\n")
    
    print("\nBacktest Completato!")
    print(open("backtest_summary.txt").read())

if __name__ == "__main__":
    main()
