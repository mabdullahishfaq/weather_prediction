{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPlFebPxPKi8ZwR80c+CMA9",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/mabdullahishfaq/weather_prediction/blob/main/save.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install streamlit -q\n",
        "!npm install -g localtunnel -q"
      ],
      "metadata": {
        "id": "yDfHUeugs8yo"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.metrics import mean_absolute_error, r2_score\n",
        "\n",
        "st.set_page_config(page_title=\"Weather Predictor\", page_icon=\"🌤️\", layout=\"wide\")\n",
        "\n",
        "st.title(\"🌤️ Weather Prediction using Machine Learning\")\n",
        "st.markdown(\"This app uses a *Random Forest* model to predict weather variables from the Seattle Weather dataset.\")\n",
        "\n",
        "@st.cache_data\n",
        "def load_data():\n",
        "    df = pd.read_csv(\"seattle-weather.csv\")\n",
        "    numeric_df = df.select_dtypes(include=[np.number])\n",
        "    return df, numeric_df\n",
        "\n",
        "df_raw, df = load_data()\n",
        "\n",
        "with st.sidebar:\n",
        "    st.header(\"⚙️ Settings\")\n",
        "    target = st.selectbox(\n",
        "        \"Select target to predict:\",\n",
        "        df.columns.tolist(),\n",
        "        index=df.columns.tolist().index(\"temp_max\") if \"temp_max\" in df.columns else 0\n",
        "    )\n",
        "    n_estimators = st.slider(\"Number of trees\", 50, 300, 100, 50)\n",
        "    test_size = st.slider(\"Test set size (%)\", 10, 40, 20, 5)\n",
        "    st.markdown(\"---\")\n",
        "    st.caption(\"Built with Streamlit & scikit-learn\")\n",
        "\n",
        "features = [col for col in df.columns if col != target]\n",
        "\n",
        "with st.expander(\"📊 Dataset Preview\", expanded=True):\n",
        "    col1, col2, col3 = st.columns(3)\n",
        "    col1.metric(\"Total Rows\", df_raw.shape[0])\n",
        "    col2.metric(\"Features Used\", len(features))\n",
        "    col3.metric(\"Target\", target)\n",
        "    st.dataframe(df_raw.head(10), use_container_width=True)\n",
        "\n",
        "st.markdown(f\"*Predicting:* {target} &nbsp;|&nbsp; *Using:* {', '.join(f'{f}' for f in features)}\")\n",
        "\n",
        "X = df[features].values\n",
        "y = df[target].values\n",
        "\n",
        "mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))\n",
        "X, y = X[mask], y[mask]\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(\n",
        "    X, y, test_size=test_size / 100, random_state=42\n",
        ")\n",
        "scaler = StandardScaler()\n",
        "X_train_s = scaler.fit_transform(X_train)\n",
        "X_test_s = scaler.transform(X_test)\n",
        "\n",
        "st.markdown(\"---\")\n",
        "if st.button(\"🚀 Train Model\", type=\"primary\", use_container_width=True):\n",
        "    with st.spinner(\"Training Random Forest...\"):\n",
        "        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)\n",
        "        model.fit(X_train_s, y_train)\n",
        "        y_pred = model.predict(X_test_s)\n",
        "\n",
        "        mae = mean_absolute_error(y_test, y_pred)\n",
        "        r2 = r2_score(y_test, y_pred)\n",
        "\n",
        "        st.session_state.update({\n",
        "            'model': model, 'scaler': scaler,\n",
        "            'mae': mae, 'r2': r2,\n",
        "            'features': features, 'target': target,\n",
        "            'y_test': y_test, 'y_pred': y_pred,\n",
        "            'trained': True\n",
        "        })\n",
        "\n",
        "if st.session_state.get('trained'):\n",
        "    mae = st.session_state['mae']\n",
        "    r2 = st.session_state['r2']\n",
        "    y_test = st.session_state['y_test']\n",
        "    y_pred = st.session_state['y_pred']\n",
        "\n",
        "    st.success(\"✅ Model trained successfully!\")\n",
        "\n",
        "    c1, c2, c3 = st.columns(3)\n",
        "    c1.metric(\"Mean Absolute Error\", f\"{mae:.3f}\")\n",
        "    c2.metric(\"R² Score\", f\"{r2:.3f}\")\n",
        "    c3.metric(\"Accuracy (approx)\", f\"{max(0, r2)*100:.1f}%\")\n",
        "\n",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n",
        "    fig.patch.set_facecolor('#0e1117')\n",
        "\n",
        "    for ax in [ax1, ax2]:\n",
        "        ax.set_facecolor('#1e2129')\n",
        "        ax.tick_params(colors='white')\n",
        "        ax.xaxis.label.set_color('white')\n",
        "        ax.yaxis.label.set_color('white')\n",
        "        ax.title.set_color('white')\n",
        "        for spine in ax.spines.values():\n",
        "            spine.set_edgecolor('#444')\n",
        "\n",
        "    ax1.scatter(y_test, y_pred, alpha=0.5, color='#4fc3f7', s=20)\n",
        "    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())\n",
        "    ax1.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect prediction')\n",
        "    ax1.set_xlabel(\"Actual\")\n",
        "    ax1.set_ylabel(\"Predicted\")\n",
        "    ax1.set_title(\"Predicted vs Actual\")\n",
        "    ax1.legend(facecolor='#1e2129', labelcolor='white')\n",
        "    ax1.grid(True, alpha=0.2)\n",
        "\n",
        "    model = st.session_state['model']\n",
        "    feats = st.session_state['features']\n",
        "    importance = model.feature_importances_\n",
        "    idx = np.argsort(importance)\n",
        "    colors = plt.cm.Blues(np.linspace(0.4, 1.0, len(feats)))\n",
        "    ax2.barh([feats[i] for i in idx], importance[idx], color=colors)\n",
        "    ax2.set_xlabel(\"Importance\")\n",
        "    ax2.set_title(\"Feature Importance\")\n",
        "    ax2.grid(True, alpha=0.2, axis='x')\n",
        "\n",
        "    plt.tight_layout()\n",
        "    st.pyplot(fig)\n",
        "\n",
        "    st.markdown(\"---\")\n",
        "    st.subheader(\"🔮 Make a Custom Prediction\")\n",
        "\n",
        "    inputs = []\n",
        "    cols = st.columns(len(feats))\n",
        "    for i, feat in enumerate(feats):\n",
        "        with cols[i]:\n",
        "            mn_v = float(df[feat].min())\n",
        "            mx_v = float(df[feat].max())\n",
        "            mean_v = float(df[feat].mean())\n",
        "            val = st.slider(feat, mn_v, mx_v, mean_v, key=f\"sl_{i}\")\n",
        "            inputs.append(val)\n",
        "\n",
        "    if st.button(\"🌡️ Predict Now\", type=\"secondary\", use_container_width=True):\n",
        "        scaled = st.session_state['scaler'].transform([inputs])\n",
        "        pred = st.session_state['model'].predict(scaled)[0]\n",
        "        st.markdown(f\"\"\"\n",
        "        <div style=\"text-align:center;padding:30px;background:linear-gradient(135deg,#1a6b9a,#764ba2);border-radius:16px;margin-top:16px;\">\n",
        "            <h3 style=\"color:#cce5ff;margin:0;\">Predicted {st.session_state['target']}</h3>\n",
        "            <h1 style=\"font-size:72px;color:white;margin:10px 0;\">{pred:.2f}</h1>\n",
        "            <p style=\"color:#ddd;margin:0;\">Estimated error: ± {st.session_state['mae']:.2f}</p>\n",
        "        </div>\n",
        "        \"\"\", unsafe_allow_html=True)\n",
        "else:\n",
        "    st.info(\"👆 Click *Train Model* above to get started.\")"
      ],
      "metadata": {
        "id": "8sj-51H_xF8r"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "%%writefile seattle-weather.csv\n",
        "date,precipitation,temp_max,temp_min,wind,weather\n",
        "2012-01-01,0.0,12.8,5.0,4.7,drizzle\n",
        "2012-01-02,10.9,10.6,2.8,4.5,rain\n",
        "2012-01-03,0.8,11.7,7.2,2.3,rain\n",
        "2012-01-04,20.3,12.2,5.6,4.7,rain\n",
        "2012-01-05,1.3,8.9,2.8,6.1,rain\n",
        "2012-01-06,2.5,4.4,2.2,2.2,rain\n",
        "2012-01-07,0.0,7.2,2.8,2.3,rain\n",
        "2012-01-08,0.0,10.0,2.8,2.0,sun\n",
        "2012-01-09,4.3,9.4,5.0,3.4,rain\n",
        "2012-01-10,1.0,6.1,0.6,3.4,rain\n",
        "2012-01-11,0.0,6.1,−1.1,5.1,sun\n",
        "2012-01-12,0.0,6.1,−1.7,1.9,sun\n",
        "2012-01-13,0.0,5.0,−2.8,1.3,sun\n",
        "2012-01-14,4.1,4.4,0.6,5.3,snow\n",
        "2012-01-15,5.3,1.1,−3.3,3.2,snow\n",
        "2012-01-16,2.5,1.7,−2.8,5.0,snow\n",
        "2012-01-17,8.1,3.3,0.0,5.6,snow\n",
        "2012-01-18,19.8,0.0,−2.8,2.1,snow\n",
        "2012-01-19,15.2,−1.1,−2.8,1.3,snow\n",
        "2012-01-20,13.5,7.2,−1.1,2.3,snow"
      ],
      "metadata": {
        "id": "DnrPCQESxkBH"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import subprocess, time\n",
        "\n",
        "process = subprocess.Popen(\n",
        "    ['streamlit', 'run', 'app.py',\n",
        "     '--server.port', '8501',\n",
        "     '--server.headless', 'true'],\n",
        "    stdout=subprocess.PIPE,\n",
        "    stderr=subprocess.PIPE\n",
        ")\n",
        "time.sleep(5)\n",
        "print(\"✅ Streamlit is running!\")"
      ],
      "metadata": {
        "id": "783FP3oQx0DR"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!npx localtunnel --port 8501"
      ],
      "metadata": {
        "id": "-EmW37s-yHHn"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import urllib.request\n",
        "print(urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())"
      ],
      "metadata": {
        "id": "Z9utuTHuyL7x"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}