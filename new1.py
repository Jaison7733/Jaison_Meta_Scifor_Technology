import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Set up the app
st.title("Random Data Chart")

# Create a slider to select the number of data points
num_points = st.slider("Number of Data Points", 10, 100)

# Generate random data
x = np.arange(num_points)
y = np.random.randn(num_points)

# Create the line chart
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_title("Random Data")

# Display the chart in Streamlit
st.pyplot(fig)