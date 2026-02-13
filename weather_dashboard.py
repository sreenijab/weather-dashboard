import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")
st.title("🌦 Live Dynamic Weather Monitoring")

# ----------------------------
# 1️⃣ Train Model Once
# ----------------------------
np.random.seed(42)
data_size = 1000

temp = np.random.normal(30, 5, data_size)
humidity = np.random.normal(70, 15, data_size)
pressure = np.random.normal(1005, 7, data_size)

rain = []
for i in range(data_size):
    if humidity[i] > 75 and pressure[i] < 1005:
        rain.append(1)
    else:
        rain.append(0)

df = pd.DataFrame({
    "Temp": temp,
    "Humidity": humidity,
    "Pressure": pressure,
    "Rain": rain
})

X = df[['Temp', 'Humidity', 'Pressure']]
y = df['Rain']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# ----------------------------
# 2️⃣ Live Simulation Section
# ----------------------------

if "live_data" not in st.session_state:
    st.session_state.live_data = pd.DataFrame(
        columns=["Temperature", "Humidity", "Pressure", "Rain Prediction"]
    )

placeholder = st.empty()

for i in range(100):   # 100 fast updates
    
    new_temp = round(np.random.normal(30, 5), 2)
    new_humidity = round(np.random.normal(70, 15), 2)
    new_pressure = round(np.random.normal(1005, 7), 2)

    input_data = pd.DataFrame([[new_temp, new_humidity, new_pressure]],
                              columns=["Temp", "Humidity", "Pressure"])

    prediction = model.predict(input_data)[0]

    rain_status = "Rain Expected 🌧" if prediction == 1 else "No Rain ☀"

    new_row = pd.DataFrame([[new_temp, new_humidity, new_pressure, rain_status]],
                           columns=["Temperature", "Humidity", "Pressure", "Rain Prediction"])

    st.session_state.live_data = pd.concat(
        [st.session_state.live_data, new_row], ignore_index=True
    )

    with placeholder.container():
        st.subheader("📊 Live Weather Values")

        st.dataframe(st.session_state.live_data.tail(10), use_container_width=True)

    time.sleep(0.5)   # Faster updates (0.5 seconds)
