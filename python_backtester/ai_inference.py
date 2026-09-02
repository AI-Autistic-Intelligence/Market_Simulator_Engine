import os
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, BatchNormalization

def parse_mqh_weights(filepath):
    print(f"Parsing {filepath}...")
    with open(filepath, 'r') as f:
        content = f.read()

    # Match layer name, shape and array values
    # e.g. // Layer: lstm_LSTM1, Shape: [11, 384]
    # double lstm_LSTM1_w0[4224] = { ... };
    
    pattern = re.compile(r'// Layer: (.*?), Shape: \[(.*?)\]\s+double \w+\[.*?\] = \{(.*?)\};', re.DOTALL)
    matches = pattern.findall(content)
    
    weights_dict = {}
    for layer_name, shape_str, values_str in matches:
        shape = tuple(map(int, shape_str.split(',')))
        values = [float(v.strip()) for v in values_str.split(',') if v.strip()]
        
        arr = np.array(values, dtype=np.float32).reshape(shape)
        
        if layer_name not in weights_dict:
            weights_dict[layer_name] = []
        weights_dict[layer_name].append(arr)
        
    return weights_dict

def build_model(weights_dict):
    WINDOW_SIZE = 30
    LSTM_UNITS = 96
    
    inp = Input(shape=(WINDOW_SIZE, 11), name="input_1")
    lstm1 = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_LSTM1")(inp)
    bn1 = BatchNormalization(name="batch_normalization_BatchNormalization1")(lstm1)
    
    lstm2 = LSTM(LSTM_UNITS, return_sequences=False, name="lstm_LSTM2")(bn1)
    bn2 = BatchNormalization(name="batch_normalization_BatchNormalization2")(lstm2)
    
    d1 = Dense(32, activation='relu', name="dense_Dense1")(bn2)
    d2 = Dense(16, activation='relu', name="dense_Dense2")(d1)
    
    action_out = Dense(2, activation='softmax', name="actionOutput")(d2)
    
    model = Model(inputs=inp, outputs=action_out)
    
    # Set weights
    for layer in model.layers:
        if layer.name in weights_dict:
            layer.set_weights(weights_dict[layer.name])
            
    return model

def calculate_indicators(df):
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # EMA 9 & 21
    df['EMA9'] = df['Close'].ewm(span=9, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    
    # BB 20
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_UP'] = sma20 + (2 * std20)
    df['BB_DN'] = sma20 - (2 * std20)
    
    # ATR 100 for trading logic
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = df[['High', 'Low']].join(high_low.rename('HL')).join(high_close.rename('HC')).join(low_close.rename('LC'))
    tr['TR'] = tr[['HL', 'HC', 'LC']].max(axis=1)
    df['ATR'] = tr['TR'].rolling(window=100).mean()
    
    df = df.bfill()
    return df
