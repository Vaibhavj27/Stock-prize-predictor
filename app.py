import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
from datetime import date
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model
model = load_model('Stock Predictions Model.keras')

# App header
st.header('📈 Stock Market Predictor')

# User input
stock = st.text_input('Enter Stock Symbol (e.g., AAPL, GOOG)', 'GOOG')
start = '2015-01-01'
end = str(date.today())

# Load stock data
data = yf.download(stock, start, end)

# Display data
st.subheader('Raw Stock Data')
st.write(data.tail())

# Train-test split
data_train = pd.DataFrame(data.Close[0:int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80):])

# Preprocessing
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

past_100_days = data_train.tail(100)
data_test_combined = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scaled = scaler.fit_transform(data_test_combined)

# Plot moving average
st.subheader('📊 50-Day Moving Average')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label='50-Day MA')
plt.plot(data.Close, 'g', label='Close Price')
plt.legend()
st.pyplot(fig1)

# Prepare data for prediction
x = []
y = []
for i in range(100, data_test_scaled.shape[0]):
    x.append(data_test_scaled[i - 100:i])
    y.append(data_test_scaled[i, 0])

x = np.array(x)
y = np.array(y)

# Predict
predict = model.predict(x)
predict = scaler.inverse_transform(predict)
y = scaler.inverse_transform(y.reshape(-1, 1))

mae = mean_absolute_error(y, predict)
mse = mean_squared_error(y, predict)
rmse = np.sqrt(mse)
r2 = r2_score(y, predict)

st.subheader('📊 Model Performance Metrics')
st.metric(label="MAE", value=f"{mae:.2f}")
st.metric(label="RMSE", value=f"{rmse:.2f}")
st.metric(label="R² Score", value=f"{r2:.4f}")
tolerance = 0.1  # 10%
accurate_preds = np.abs(predict - y) <= (tolerance * y)
accuracy_percent = np.mean(accurate_preds) * 100

st.write(f"✅ Custom Accuracy : {accuracy_percent:.2f}%")


# Show prediction chart
st.subheader('📈 Predicted vs Actual Prices')
fig2 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# Show next day prediction
last_100_days = data_test_combined[-100:].values
last_100_scaled = scaler.transform(last_100_days)
last_100_scaled = np.reshape(last_100_scaled, (1, 100, 1))

next_day_prediction = model.predict(last_100_scaled)
next_day_prediction = scaler.inverse_transform(next_day_prediction)[0][0]


st.subheader('🔮 Next Day Predicted Price')
st.success(f"The predicted closing price for {stock.upper()} on the next day is: {next_day_prediction:.2f}")
