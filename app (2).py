import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="Weather Predictor", page_icon="🌤️", layout="wide")

st.title("🌤️ Weather Prediction using Machine Learning")
st.markdown("This app uses a *Random Forest* model to predict weather variables from the Seattle Weather dataset.")

@st.cache_data
def load_data():
    df = pd.read_csv("seattle-weather.csv")
    numeric_df = df.select_dtypes(include=[np.number])
    return df, numeric_df

df_raw, df = load_data()

with st.sidebar:
    st.header("⚙️ Settings")
    target = st.selectbox(
        "Select target to predict:",
        df.columns.tolist(),
        index=df.columns.tolist().index("temp_max") if "temp_max" in df.columns else 0
    )
    n_estimators = st.slider("Number of trees", 50, 300, 100, 50)
    test_size = st.slider("Test set size (%)", 10, 40, 20, 5)
    st.markdown("---")
    st.caption("Built with Streamlit & scikit-learn")

features = [col for col in df.columns if col != target]

with st.expander("📊 Dataset Preview", expanded=True):
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df_raw.shape[0])
    col2.metric("Features Used", len(features))
    col3.metric("Target", target)
    st.dataframe(df_raw.head(10), use_container_width=True)

st.markdown(f"*Predicting:* {target} &nbsp;|&nbsp; *Using:* {', '.join(f'{f}' for f in features)}")

X = df[features].values
y = df[target].values

mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size / 100, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

st.markdown("---")
if st.button("🚀 Train Model", type="primary", use_container_width=True):
    with st.spinner("Training Random Forest..."):
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.session_state.update({
            'model': model, 'scaler': scaler,
            'mae': mae, 'r2': r2,
            'features': features, 'target': target,
            'y_test': y_test, 'y_pred': y_pred,
            'trained': True
        })

if st.session_state.get('trained'):
    mae = st.session_state['mae']
    r2 = st.session_state['r2']
    y_test = st.session_state['y_test']
    y_pred = st.session_state['y_pred']

    st.success("✅ Model trained successfully!")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Absolute Error", f"{mae:.3f}")
    c2.metric("R² Score", f"{r2:.3f}")
    c3.metric("Accuracy (approx)", f"{max(0, r2)*100:.1f}%")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor('#0e1117')

    for ax in [ax1, ax2]:
        ax.set_facecolor('#1e2129')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

    ax1.scatter(y_test, y_pred, alpha=0.5, color='#4fc3f7', s=20)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    ax1.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')
    ax1.set_xlabel("Actual")
    ax1.set_ylabel("Predicted")
    ax1.set_title("Predicted vs Actual")
    ax1.legend(facecolor='#1e2129', labelcolor='white')
    ax1.grid(True, alpha=0.2)

    model = st.session_state['model']
    feats = st.session_state['features']
    importance = model.feature_importances_
    idx = np.argsort(importance)
    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(feats)))
    ax2.barh([feats[i] for i in idx], importance[idx], color=colors)
    ax2.set_xlabel("Importance")
    ax2.set_title("Feature Importance")
    ax2.grid(True, alpha=0.2, axis='x')

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("🔮 Make a Custom Prediction")

    inputs = []
    cols = st.columns(len(feats))
    for i, feat in enumerate(feats):
        with cols[i]:
            mn_v = float(df[feat].min())
            mx_v = float(df[feat].max())
            mean_v = float(df[feat].mean())
            val = st.slider(feat, mn_v, mx_v, mean_v, key=f"sl_{i}")
            inputs.append(val)

    if st.button("🌡️ Predict Now", type="secondary", use_container_width=True):
        scaled = st.session_state['scaler'].transform([inputs])
        pred = st.session_state['model'].predict(scaled)[0]
        st.markdown(f"""
        <div style="text-align:center;padding:30px;background:linear-gradient(135deg,#1a6b9a,#764ba2);border-radius:16px;margin-top:16px;">
            <h3 style="color:#cce5ff;margin:0;">Predicted {st.session_state['target']}</h3>
            <h1 style="font-size:72px;color:white;margin:10px 0;">{pred:.2f}</h1>
            <p style="color:#ddd;margin:0;">Estimated error: ± {st.session_state['mae']:.2f}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("👆 Click *Train Model* above to get started.")
