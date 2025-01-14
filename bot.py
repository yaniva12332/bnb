import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator

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
def fetch_real_time_data():
    reserves = contract.functions.getReserves().call()
    price = reserves[1] / reserves[0]
    return round(price, 6)

def fetch_historical_data():
    data = pd.read_csv('historical_pancakeswap_data.csv')
    data['RSI'] = RSIIndicator(close=data['Close']).rsi()
    data['MACD'] = MACD(close=data['Close']).macd_diff()
    data['SMA'] = SMAIndicator(close=data['Close']).sma_indicator()
    data.fillna(method='bfill', inplace=True)
    return data

# =======================
# שלב 2: הכנת הנתונים
# =======================
def preprocess_data(data):
    X = data[['Close', 'RSI', 'MACD', 'SMA']].values
    y = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    return X, y

# =======================
# שלב 3: בניית מודל LSTM
# =======================
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =======================
# שלב 4: שילוב מודלים (Ensemble)
# =======================
def build_ensemble_models(X, y):
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
    xgb_model = XGBClassifier(n_estimators=200, learning_rate=0.1)

    rf_model.fit(X, y)
    xgb_model.fit(X, y)

    return rf_model, xgb_model

# =======================
# שלב 5: חיזוי תוצאה
# =======================
def predict_next_outcome(data):
    X, y = preprocess_data(data)
    X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

    lstm_model = build_lstm_model((X_reshaped.shape[1], X_reshaped.shape[2]))
    lstm_model.fit(X_reshaped, y, epochs=10, batch_size=32, verbose=0)

    rf_model, xgb_model = build_ensemble_models(X, y)

    lstm_pred = lstm_model.predict(X_reshaped[-1].reshape(1, 1, X.shape[1]))
    rf_pred = rf_model.predict(X[-1].reshape(1, -1))
    xgb_pred = xgb_model.predict(X[-1].reshape(1, -1))

    # חיזוי משולב
    final_prediction = np.round((lstm_pred + rf_pred + xgb_pred) / 3).astype(int)

    return "UP" if final_prediction[0][0] == 1 else "DOWN"

# =======================
# שלב 6: הפעלת הבוט
# =======================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = fetch_historical_data()
    prediction = predict_next_outcome(data)
    await update.message.reply_text(f"התחזית להגרלה הבאה ב-PancakeSwap היא: {prediction}")

async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", predict))
    print("✅ Bot is running!")
    await app.run_polling()

if __name__ == "__main__":
    try:
        asyncio.run(start_bot())
    except RuntimeError as e:
        if "This event loop is already running" in str(e):
            loop = asyncio.get_event_loop()
            loop.create_task(start_bot())
            loop.run_forever()
        else:
            raise