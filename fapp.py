import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Weather Predictor", layout="wide")
st.title("🌤️ Weather Predictor")

# Load your data
try:
    df = pd.read_csv("sample2.csv")
    st.success("✅ Data loaded successfully!")
except:
    st.error("❌ Please upload seattle-weather.csv to your repository")
    st.stop()

st.subheader("📊 Your Data")
st.dataframe(df.head())

# Let user select what to predict
target = st.selectbox("What do you want to predict?", df.columns.tolist())
features = [col for col in df.columns if col != target]

st.write(f"**Predicting:** {target}")
st.write(f"**Using factors:** {', '.join(features)}")

# Prepare data
X = df[features].values
y = df[target].values

# Remove any rows with missing values
if pd.isnull(X).any() or pd.isnull(y).any():
    st.warning("Removing rows with missing values...")
    mask = ~(pd.isnull(X).any(axis=1) | pd.isnull(y))
    X = X[mask]
    y = y[mask]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model button
if st.button("🚀 Train Model", type="primary"):
    with st.spinner("Training Random Forest model..."):
        # Create and train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store in session
        st.session_state['model'] = model
        st.session_state['scaler'] = scaler
        st.session_state['mae'] = mae
        st.session_state['r2'] = r2
        st.session_state['features'] = features
        st.session_state['target'] = target
        st.session_state['trained'] = True
        
        # Show results
        st.success("✅ Model trained successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Mean Absolute Error", f"{mae:.2f}")
        with col2:
            st.metric("R² Score", f"{r2:.3f}")
        
        # Plot predictions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot
        ax1.scatter(y_test, y_pred, alpha=0.5, color='blue')
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel("Actual")
        ax1.set_ylabel("Predicted")
        ax1.set_title("Predictions vs Actual")
        ax1.grid(True, alpha=0.3)
        
        # Feature importance
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        ax2.barh(range(len(features)), importance[indices])
        ax2.set_yticks(range(len(features)))
        ax2.set_yticklabels([features[i] for i in indices])
        ax2.set_xlabel("Importance")
        ax2.set_title("Feature Importance")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)

# Prediction section
if 'trained' in st.session_state:
    st.subheader("🔮 Make a Prediction")
    
    # Create input fields
    inputs = []
    cols = st.columns(len(st.session_state['features']))
    
    for i, feature in enumerate(st.session_state['features']):
        with cols[i % len(cols)]:
            mean_val = float(df[feature].mean())
            val = st.number_input(f"{feature}", value=mean_val, step=0.1, key=f"input_{i}")
            inputs.append(val)
    
    if st.button("🌡️ Predict Now", type="secondary"):
        # Scale and predict
        input_scaled = st.session_state['scaler'].transform([inputs])
        prediction = st.session_state['model'].predict(input_scaled)[0]
        
        # Show result
        st.markdown(f"""
        <div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px;">
            <h2 style="color: white;">📈 Predicted Value</h2>
            <h1 style="font-size: 64px; color: white;">{prediction:.2f}</h1>
            <p style="color: white;">{st.session_state['target']}</p>
            <hr style="border-color: rgba(255,255,255,0.3);">
            <p style="color: white; font-size: 14px;">Model Error: ± {st.session_state['mae']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👈 Click 'Train Model' to start")

# Footer
st.markdown("---")
st.caption("Built with Streamlit & Random Forest | Weather Prediction App")
