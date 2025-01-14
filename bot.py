import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# =======================
# הגדרות ראשוניות
# =======================
TELEGRAM_TOKEN = "7808322309:AAFNGvGcI8Gk_PGmDMmtLcCbZhVuahHgiZI"

bsc = "https://bsc-dataseed.binance.org/"
web3 = Web3(Web3.HTTPProvider(bsc))
prediction_contract_address = web3.to_checksum_address('0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA')

contract = web3.eth.contract(address=prediction_contract_address, abi=[
    {
        "constant": True,
        "inputs": [],
        "name": "getRoundData",
        "outputs": [
            {"name": "epoch", "type": "uint256"},
            {"name": "bullAmount", "type": "uint256"},
            {"name": "bearAmount", "type": "uint256"},
            {"name": "totalAmount", "type": "uint256"}
        ],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    }
])

# =======================
# שלב 1: איסוף נתונים
# =======================
def fetch_round_data():
    data = contract.functions.getRoundData().call()
    return {
        "epoch": data[0],
        "bull_amount": data[1],
        "bear_amount": data[2],
        "total_amount": data[3]
    }

# =======================
# שלב 2: עיבוד נתונים
# =======================
def preprocess_data(data):
    df = pd.DataFrame(data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    X = scaled_data[:, :-1]
    y = scaled_data[:, -1]
    return X, y, scaler

# =======================
# שלב 3: בניית מודל LSTM משופר
# =======================
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# =======================
# שלב 4: שילוב מודלים (Ensemble)
# =======================
def build_ensemble_model(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=300, max_depth=15)
    xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.05)

    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    return rf_model, xgb_model

# =======================
# שלב 5: חיזוי
# =======================
def predict_next_round(data):
    X, y, scaler = preprocess_data(data)
    lstm_model = build_lstm_model((X.shape[1], X.shape[2]))
    lstm_model.fit(X, y, epochs=50, batch_size=32, verbose=0)

    rf_model, xgb_model = build_ensemble_model(X, y)

    lstm_pred = lstm_model.predict(X[-1].reshape(1, X.shape[1], X.shape[2]))
    rf_pred = rf_model.predict(X[-1].reshape(1, -1))
    xgb_pred = xgb_model.predict(X[-1].reshape(1, -1))

    final_prediction = (lstm_pred + rf_pred + xgb_pred) / 3
    return "Bull" if final_prediction > 0.5 else "Bear"

# =======================
# שלב 6: הגדרת הבוט
# =======================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    data = [fetch_round_data()]
    prediction = predict_next_round(data)
    await update.message.reply_text(f"הניבוי להגרלה הבאה: {prediction}")

async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", predict))
    print("✅ Bot is running!")
    await app.run_polling()

if __name__ == "__main__":
    asyncio.run(start_bot())