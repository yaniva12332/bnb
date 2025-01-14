import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =======================
# הגדרות ראשוניות
# =======================
TELEGRAM_TOKEN = "7808322309:AAFNGvGcI8Gk_PGmDMmtLcCbZhVuahHgiZI"
bsc = "https://bsc-dataseed.binance.org/"
web3 = Web3(Web3.HTTPProvider(bsc))

# כתובת חוזה Prediction של PancakeSwap
prediction_address = web3.to_checksum_address('0x0eD7e52944161450477ee417DE9Cd3a859b14fD0')

# ABI של חוזה PancakeSwap Prediction
prediction_abi = [
    {
        "inputs": [],
        "name": "currentEpoch",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "uint256", "name": "_epoch", "type": "uint256"}],
        "name": "getRound",
        "outputs": [
            {"internalType": "uint256", "name": "epoch", "type": "uint256"},
            {"internalType": "uint256", "name": "lockPrice", "type": "uint256"},
            {"internalType": "uint256", "name": "closePrice", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# התחברות לחוזה Prediction
prediction_contract = web3.eth.contract(address=prediction_address, abi=prediction_abi)

# =======================
# שלב 1: איסוף נתונים מהחוזה Prediction
# =======================
def fetch_prediction_data():
    current_epoch = prediction_contract.functions.currentEpoch().call()
    data = []
    
    # איסוף נתונים מ-50 סיבובים קודמים
    for epoch in range(current_epoch - 50, current_epoch):
        round_data = prediction_contract.functions.getRound(epoch).call()
        lock_price = round_data[1]
        close_price = round_data[2]
        result = 1 if close_price > lock_price else 0  # 1 = עלייה, 0 = ירידה
        data.append([lock_price, close_price, result])
    
    return np.array(data)

# =======================
# שלב 2: עיבוד נתונים ואימון מודל
# =======================
def train_model():
    data = fetch_prediction_data()
    X = data[:, :2]  # lock_price ו-close_price
    y = data[:, 2]   # תוצאה (עלייה/ירידה)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler

# =======================
# שלב 3: חיזוי התוצאה הבאה
# =======================
def predict_next_round(model, scaler):
    current_epoch = prediction_contract.functions.currentEpoch().call()
    round_data = prediction_contract.functions.getRound(current_epoch - 1).call()
    lock_price = round_data[1]
    close_price = round_data[2]
    
    X_new = scaler.transform([[lock_price, close_price]])
    prediction = model.predict(X_new)
    
    return "🔼 עלייה" if prediction[0] == 1 else "🔽 ירידה"

# =======================
# שלב 4: הגדרת הפקודה בבוט
# =======================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model, scaler = train_model()
    prediction = predict_next_round(model, scaler)
    
    await update.message.reply_text(f"התחזית לסיבוב הבא: {prediction}")

# =======================
# שלב 5: הפעלת הבוט
# =======================
async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", predict))
    
    print("✅ Bot is running!")
    await app.run_polling()

if __name__ == "__main__":
    import platform
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.get_event_loop()
    loop.create_task(start_bot())
    loop.run_forever()