!pip install streamlit
import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.title("🌦 Weather Rain Prediction Dashboard")

# ---------------------------
# 1️⃣ Load Dataset
# ---------------------------
data = pd.read_csv("weather.csv")

X = data[['Temp', 'Humidity', 'Pressure']]
y = data['Rain']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# 2️⃣ Train Model
# ---------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.sidebar.success(f"Model Accuracy: {round(accuracy*100,2)}%")

# ---------------------------
# 3️⃣ User Input Section
# ---------------------------
st.subheader("Enter Current Weather Parameters")

temp = st.slider("Temperature (°C)", 0, 50, 30)
humidity = st.slider("Humidity (%)", 0, 100, 70)
pressure = st.slider("Pressure (hPa)", 950, 1050, 1005)

# ---------------------------
# 4️⃣ Prediction
# ---------------------------
if st.button("Predict Weather"):

    input_data = pd.DataFrame([[temp, humidity, pressure]],
                              columns=['Temp', 'Humidity', 'Pressure'])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("🌧 Rain Expected")
    else:
        st.success("☀ No Rain Expected")