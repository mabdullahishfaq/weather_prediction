import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

st.set_page_config(page_title="Weather Predictor", layout="wide")
st.title("🌤️ Weather Predictor")

# Try to load your data
try:
    df = pd.read_csv("sample2.csv")
    st.success("✅ Loaded your data!")
except:
    st.warning("Using sample data")
    np.random.seed(42)
    df = pd.DataFrame({
        'temp_max': np.random.normal(20, 8, 1000),
        'temp_min': np.random.normal(10, 5, 1000),
        'wind': np.random.exponential(5, 1000),
    })

st.dataframe(df.head())

# Select target
target = st.selectbox("What to predict?", df.columns.tolist(), index=len(df.columns)-1)
features = [col for col in df.columns if col != target]

# Prepare data
X = df[features].values
y = df[target].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create model
model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train
if st.button("Train Model"):
    with st.spinner("Training..."):
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)
        loss, mae = model.evaluate(X_test, y_test, verbose=0)
        st.success(f"✅ Done! Error: {mae:.2f}")
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler

# Predict
if 'model' in st.session_state:
    st.subheader("Make Prediction")
    inputs = []
    cols = st.columns(len(features))
    for i, feat in enumerate(features):
        with cols[i]:
            val = st.number_input(feat, value=float(df[feat].mean()))
            inputs.append(val)
    
    if st.button("Predict"):
        result = st.session_state['model'].predict(st.session_state['scaler'].transform([inputs]))[0][0]
        st.success(f"Predicted {target}: {result:.2f}")
