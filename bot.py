import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from web3 import Web3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# =======================
# הגדרות ראשוניות
# =======================
TELEGRAM_TOKEN = '7808322309:AAFNGvGcI8Gk_PGmDMmtLcCbZhVuahHgiZI'
bsc = "https://bsc-dataseed.binance.org/"
web3 = Web3(Web3.HTTPProvider(bsc))
pair_address = web3.toChecksumAddress('0x18B2A687610328590Bc8F2e5fEdDe3b582A49cdA')

# התחברות לחוזה חכם
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

# =======================
# שלב 2: סימולציית נתונים להגרלה
# =======================
def generate_simulated_data():
    np.random.seed(42)
    data = pd.DataFrame({
        'price_change': np.random.randn(1000),
        'volume': np.random.rand(1000) * 1000,
        'indicator': np.random.choice([0, 1], 1000)
    })
    data['target'] = np.where(data['price_change'] > 0, 1, 0)
    return data

# =======================
# שלב 3: בניית מודל חיזוי
# =======================
def train_model():
    data = generate_simulated_data()
    X = data[['price_change', 'volume', 'indicator']]
    y = data['target']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=500, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# =======================
# שלב 4: חיזוי תוצאה
# =======================
def predict_lottery_outcome(model, scaler):
    current_data = np.array([[np.random.randn(), np.random.rand() * 1000, np.random.choice([0, 1])]])
    current_data_scaled = scaler.transform(current_data)
    prediction = model.predict(current_data_scaled)
    
    return "UP" if prediction[0] == 1 else "DOWN"

# =======================
# שלב 5: בוט טלגרם
# =======================
async def predict(update: Update, context: ContextTypes.DEFAULT_TYPE):
    model, scaler = train_model()
    prediction = predict_lottery_outcome(model, scaler)
    await update.message.reply_text(f"התחזית לתוצאה הבאה ב-PancakeSwap היא: {prediction}")

async def start_bot():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("predict", predict))

    print("✅ Bot is running!")
    await app.run_polling()

if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(start_bot())
        else:
            loop.run_until_complete(start_bot())
    except RuntimeError as e:
        print(f"Runtime Error: {e}")
    except Exception as e:
        print(f"Unexpected Error: {e}")