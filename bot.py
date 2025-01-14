import asyncio
import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, ADXIndicator
from ta.volatility import BollingerBands

# =======================
# הגדרות ראשוניות
# =======================
TELEGRAM_TOKEN = "7808322309:AAFNGvGcI8Gk_PGmDMmtLcCbZhVuahHgiZI"
bsc = "https://bsc-dataseed.binance.org/"
web3 = Web3(Web3.HTTPProvider(bsc))
prediction_address = web3.to_checksum_address('0x0eD7e52944161450477ee417DE9Cd3a859b14fD0')

# =======================
# ABI של החוזה החכם
# =======================
abi = [
    {"inputs":[],"payable":False,"stateMutability":"nonpayable","type":"constructor"},
    {"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"owner","type":"address"},{"indexed":True,"internalType":"address","name":"spender","type":"address"},{"indexed":False,"internalType":"uint256","name":"value","type":"uint256"}],"name":"Approval","type":"event"},
    {"anonymous":False,"inputs":[{"indexed":True,"internalType":"address","name":"sender","type":"address"},{"indexed":False,"internalType":"uint256","name":"amount0","type":"uint256"},{"indexed":False,"internalType":"uint256","name":"amount1","type":"uint256"},{"indexed":True,"internalType":"address","name":"to","type":"address"}],"name":"Burn","type":"event"},
    {"constant":True,"inputs":[],"name":"DOMAIN_SEPARATOR","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"MINIMUM_LIQUIDITY","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[],"name":"PERMIT_TYPEHASH","outputs":[{"internalType":"bytes32","name":"","type":"bytes32"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":True,"inputs":[{"internalType":"address","name":"","type":"address"},{"internalType":"address","name":"","type":"address"}],"name":"allowance","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":False,"inputs":[{"internalType":"address","name":"spender","type":"address"},{"internalType":"uint256","name":"value","type":"uint256"}],"name":"approve","outputs":[{"internalType":"bool","name":"","type":"bool"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
    {"constant":True,"inputs":[{"internalType":"address","name":"","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":False,"inputs":[{"internalType":"address","name":"to","type":"address"}],"name":"burn","outputs":[{"internalType":"uint256","name":"amount0","type":"uint256"},{"internalType":"uint256","name":"amount1","type":"uint256"}],"payable":False,"stateMutability":"nonpayable","type":"function"},
    {"constant":True,"inputs":[],"name":"getReserves","outputs":[{"internalType":"uint112","name":"_reserve0","type":"uint112"},{"internalType":"uint112","name":"_reserve1","type":"uint112"},{"internalType":"uint32","name":"_blockTimestampLast","type":"uint32"}],"payable":False,"stateMutability":"view","type":"function"},
    {"constant":False,"inputs":[{"internalType":"uint256","name":"amount0Out","type":"uint256"},{"internalType":"uint256","name":"amount1Out","type":"uint256"},{"internalType":"address","name":"to","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"swap","outputs":[],"payable":False,"stateMutability":"nonpayable","type":"function"},
    {"constant":True,"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"payable":False,"stateMutability":"view","type":"function"}
]

# =======================
# חיבור לחוזה החכם
# =======================
prediction_contract = web3.eth.contract(address=prediction_address, abi=abi)

# =======================
# משיכת מחירים בזמן אמת
# =======================
def get_crypto_prices():
    symbols = ["BNBUSDT", "BTCUSDT", "ETHUSDT"]
    prices = {}
    for symbol in symbols:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
        response = requests.get(url)
        prices[symbol] = float(response.json()['price'])
    return prices

# =======================
# ניתוח סנטימנט מחדשות
# =======================
def get_sentiment_score():
    # לדוגמה: שילוב עם API שנותן סנטימנט
    url = "https://api.alternative.me/fng/"
    response = requests.get(url)
    data = response.json()
    return int(data['data'][0]['value'])

# =======================
# איסוף נתונים מתקדם
# =======================
def fetch_prediction_data():
    current_epoch = prediction_contract.functions.currentEpoch().call()
    data = []

    for epoch in range(current_epoch - 150, current_epoch):
        round_data = prediction_contract.functions.getRound(epoch).call()
        lock_price, close_price = round_data[1], round_data[2]
        result = 1 if close_price > lock_price else 0
        crypto_prices = get_crypto_prices()
        sentiment = get_sentiment_score()
        data.append([lock_price, close_price, crypto_prices["BNBUSDT"], crypto_prices["BTCUSDT"], crypto_prices["ETHUSDT"], sentiment, result])

    df = pd.DataFrame(data, columns=["lock_price", "close_price", "bnb_price", "btc_price", "eth_price", "sentiment", "result"])

    # אינדיקטורים טכניים
    df['RSI'] = RSIIndicator(close=df['close_price']).rsi()
    df['MACD'] = MACD(close=df['close_price']).macd_diff()
    df['SMA'] = SMAIndicator(close=df['close_price']).sma_indicator()
    df['Stochastic'] = StochasticOscillator(high=df['lock_price'], low=df['close_price'], close=df['close_price']).stoch()
    df['Bollinger'] = BollingerBands(close=df['close_price']).bollinger_hband()
    df['ADX'] = ADXIndicator(high=df['lock_price'], low=df['close_price'], close=df['close_price']).adx()

    df.fillna(method='bfill', inplace=True)
    return df

# =======================
# בניית מודל Bidirectional LSTM
# =======================
def build_bidirectional_lstm(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =======================
# אימון מודלים משולבים
# =======================
def train_advanced_models():
    df = fetch_prediction_data()
    features = df[['lock_price', 'close_price', 'bnb_price', 'btc_price', 'eth_price', 'sentiment', 'RSI', 'MACD', 'SMA', 'Stochastic', 'Bollinger', 'ADX']]
    target = df['result']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(features)

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=800, max_depth=30, random_state=42)

    # XGBoost
    xgb_model = XGBClassifier(n_estimators=800, learning_rate=0.001, max_depth=25)

    # LightGBM
    lgb_model = LGBMClassifier(n_estimators=800, learning_rate=0.001, max_depth=25)

    # Stacking
    stacked_model = StackingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model)], final_estimator=lgb_model)
    stacked_model.fit(X_scaled, target)

    # Bidirectional LSTM
    lstm_model = build_bidirectional_lstm((X_scaled.shape[1], 1))
    early_stop = EarlyStopping(monitor='loss', patience=10)
    lstm_model.fit(X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1), target, epochs=200, batch_size=32, verbose=0, callbacks=[early_stop])

    return stacked_model, lstm_model, scaler

# =======================
# חיזוי מתקדם
# =======================
def advanced_prediction(stacked_model, lstm_model, scaler):
    crypto_prices = get_crypto_prices()
    sentiment = get_sentiment_score()
    X_new = scaler.transform([[crypto_prices["BNBUSDT"], crypto_prices["BTCUSDT"], crypto_prices["ETHUSDT"], sentiment, 50, 0, 0, 0, 0, 0, 0]])

    lstm_pred = lstm_model.predict(X_new.reshape(1, X_new.shape[1], 1))[0][0]
    stacked_pred = stacked_model.predict(X_new)[0]

    final_prediction = round((lstm_pred + stacked_pred) / 2)
    return "🔼 עלייה" if final_prediction == 1 else "🔽 ירידה"

# =======================
# הפעלת בוט טלגרם
# =======================
async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build>
    app.add_handler(CommandHandler("predict", predict))
    print("✅ Bot is running!")

    await app.initialize()
    await app.start()
    await app.updater.start_polling()

if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(start_bot())
    except (KeyboardInterrupt, SystemExit):
        print("⛔️ Bot stopped.")

