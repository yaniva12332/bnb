import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator, MACD

# =======================
# הגדרות ראשוניות
# =======================
TELEGRAM_TOKEN = "7808322309:AAFNGvGcI8Gk_PGmDMmtLcCbZhVuahHgiZI"
bsc = "https://bsc-dataseed.binance.org/"
web3 = Web3(Web3.HTTPProvider(bsc))
pair_address = web3.to_checksum_address('0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA')

contract = web3.eth.contract(address=pair_address, abi=[
    {
        "constant": True,
        "inputs": [],
        "name": "getReserves",
        "outputs": [
            {"name": "_reserve0", "type": "uint112"},
            {"name": "_reserve1", "type": "uint112"},
            {"name": "_blockTimestampLast", "type": "uint32"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
])

# =======================
# שלב 1: איסוף נתונים בזמן אמת
# =======================
def get_bnb_price():
    reserves = contract.functions.getReserves().call()
    price = reserves[1] / reserves[0]
    return round(price, 2)

def fetch_historical_data():
    data = yf.download('BNB-USD', period='60d', interval='5m')
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    data['SMA'] = SMAIndicator(close=data['Close'], window=14).sma_indicator()
    data['MACD'] = MACD(close=data['Close']).macd_diff()
    data.fillna(method='bfill', inplace=True)
    return data

# =======================
# שלב 2: הכנת הנתונים
# =======================
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'RSI', 'SMA', 'MACD']])
    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y), scaler

# =======================
# שלב 3: בניית מודל LSTM משופר
# =======================
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# =======================
# שלב 4: שילוב מודלים (Ensemble)
# =======================
def build_ensemble_model(X_train, y_train):
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=10)
    xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1)
    
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model

# =======================
# שלב 5: חיזוי
# =======================
def predict_next_price(data):
    X, y, scaler = preprocess_data(data)
    lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    rf_model, xgb_model = build_ensemble_model(X.reshape(X.shape[0], -1), y)
    
    lstm_pred = lstm_model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    rf_pred = rf_model.predict(X[-1].reshape(1, -1))
    xgb_pred = xgb_model.predict(X[-1].reshape(1, -1))
    
    final_prediction = (lstm_pred + rf_pred + xgb_pred) / 3
    return scaler.inverse_transform([[final_prediction[0][0], 0, 0, 0]])[0][0]

# =======================
# שלב 6: הגדרת הבוט
# =======================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = fetch_historical_data()
    predicted_price = predict_next_price(data)
    await update.message.reply_text(f"המחיר החזוי של BNB בעוד 5 דקות הוא: ${predicted_price:.2f}")

async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", predict))
    print("✅ Bot is running!")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(start_bot())
