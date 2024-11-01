import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Set the title of the dashboard
st.title("Stock Market Dashboard")

# Sidebar for user input
st.sidebar.header("User Input")

# Input for stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g. AAPL, MSFT):", "AAPL")

# Input for date range
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Load data
data = yf.download(ticker, start_date, end_date)


# Display stock data
st.subheader(f"Stock Data for {ticker}")
st.write(data)

# Plot closing price
st.subheader("Closing Price Chart")
fig, ax = plt.subplots()
ax.plot(data['Close'], label='Close Price', color='blue')
ax.set_title(f'{ticker} Closing Price')
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.legend()
st.pyplot(fig)

# Calculate moving averages
window_size_50 = 50
window_size_200 = 200
data['50_MA'] = data['Close'].rolling(window=window_size_50).mean()
data['200_MA'] = data['Close'].rolling(window=window_size_200).mean()

# Plot moving averages
st.subheader("Moving Averages")
fig_ma, ax_ma = plt.subplots()
ax_ma.plot(data['Close'], label='Close Price', color='blue')
ax_ma.plot(data['50_MA'], label='50-Day MA', color='orange')
ax_ma.plot(data['200_MA'], label='200-Day MA', color='red')
ax_ma.set_title(f'{ticker} Moving Averages')
ax_ma.set_xlabel('Date')
ax_ma.set_ylabel('Price (USD)')
ax_ma.legend()
st.pyplot(fig_ma)

# Downloadable report
csv = data.to_csv(index=True).encode('utf-8')
st.download_button(
    label="Download CSV",
    data=csv,
    file_name=f"{ticker}_data.csv",
    mime='text/csv',
)
