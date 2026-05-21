# weather_prediction
this project apply ann to make weather prediction
## Project Description: Weather Prediction using Artificial Neural Network (ANN)

### Project Title
**Weather Prediction System: Next Day Maximum Temperature Forecasting using Deep Learning**

---

### 1. Overview

This project implements a **Regression-based Artificial Neural Network (ANN)** to predict the **next day's maximum temperature** based on current weather parameters. The system generates a synthetic weather dataset, preprocesses it, trains a deep learning model, and evaluates its performance using Mean Absolute Error (MAE).

---

### 2. Problem Statement

Weather prediction is a complex task due to the non-linear relationships between various meteorological parameters. Traditional statistical methods often struggle to capture these complex patterns. This project aims to:

- **Predict** the next day's maximum temperature using current weather data
- **Demonstrate** the application of deep learning for time-series forecasting
- **Handle** synthetic weather data to simulate real-world scenarios

---

### 3. Dataset Description

The project generates a **synthetic dataset** containing **50,000 samples** with the following features:

| Feature | Description | Unit |
|---------|-------------|------|
| `Humidity` | Relative humidity | Percentage (%) |
| `WindSpeed` | Wind speed | m/s |
| `Pressure` | Atmospheric pressure | hPa |
| `CurrentMaxTemp` | Current day's maximum temperature | °C |
| `CurrentMinTemp` | Current day's minimum temperature | °C |
| `NextDayMaxTemp` | **Target variable** - Next day's maximum temperature | °C |

**Data Generation Logic:**
- Seasonal patterns are incorporated using sine functions
- Random noise is added to simulate real-world variations
- Realistic relationships are maintained (e.g., temperature affects pressure and humidity)

---

### 4. Methodology

#### 4.1 Data Preprocessing
- **Train-Test Split:** 80% training, 20% testing
- **Feature Scaling:** StandardScaler (z-score normalization) to standardize features
- **Random State:** 42 for reproducible results

#### 4.2 Neural Network Architecture

```
Input Layer (5 features)
       ↓
Dense Layer 1: 64 neurons (ReLU activation)
       ↓
Dropout Layer: 20% (prevents overfitting)
       ↓
Dense Layer 2: 64 neurons (ReLU activation)
       ↓
Dropout Layer: 20%
       ↓
Dense Layer 3: 32 neurons (ReLU activation)
       ↓
Output Layer: 1 neuron (linear activation for regression)
```

#### 4.3 Training Configuration
- **Optimizer:** Adam (adaptive learning rate)
- **Loss Function:** Mean Squared Error (MSE)
- **Evaluation Metric:** Mean Absolute Error (MAE)
- **Batch Size:** 32
- **Epochs:** 20
- **Validation Split:** 10% of training data

---

### 5. Key Features

| Feature | Description |
|---------|-------------|
| **Synthetic Data Generation** | Creates realistic weather data with seasonal patterns |
| **Deep Learning Model** | 4-layer neural network with dropout regularization |
| **Regression Task** | Predicts continuous temperature values |
| **Visualization** | Scatter plot comparing actual vs predicted values |
| **Model Persistence** | Saves trained model as HDF5 file |

---

### 6. Expected Outputs

1. **Dataset Generation Confirmation**
   ```
   Success! Generated 'sample.csv' with 50000 rows.
   ```

2. **Model Architecture Summary**
   - Total parameters: ~6,657
   - Trainable parameters: ~6,657

3. **Training Progress**
   - Loss and MAE for each epoch
   - Validation metrics

4. **Evaluation Results**
   - Mean Absolute Error on test set (typically around 2.0°C)

5. **Visualization**
   - Scatter plot showing predicted vs actual temperatures
   - Diagonal line representing perfect predictions

6. **Saved Model**
   - `weather_ann_model.h5` file

---

### 7. Applications

This project can be extended for:

- **Real-time weather forecasting** using historical weather data
- **Agricultural planning** (frost prediction, crop management)
- **Energy demand forecasting** (heating/cooling load prediction)
- **Disaster preparedness** (extreme temperature event warning)
- **Smart building management** (HVAC optimization)

---

### 8. Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **NumPy** | Numerical computations and random data generation |
| **Pandas** | Data manipulation and CSV handling |
| **Scikit-learn** | Data splitting and feature scaling |
| **TensorFlow/Keras** | Neural network building and training |
| **Matplotlib** | Data visualization |

---

### 9. Limitations & Future Improvements

| Limitation | Future Improvement |
|------------|---------------------|
| Synthetic data (not real-world) | Use real historical weather data |
| Only 5 input features | Include more features (cloud cover, precipitation, etc.) |
| Simple feedforward architecture | Use LSTM for time-series forecasting |
| No hyperparameter tuning | Implement Grid Search or Random Search |
| Single-step prediction | Multi-step forecasting |

---

### 10. Conclusion

This project successfully demonstrates the application of **Artificial Neural Networks** for **weather temperature prediction**. The model achieves reasonable accuracy (MAE ~2°C) on synthetic data, showcasing the potential of deep learning in meteorological forecasting. The modular design allows easy adaptation for real-world weather data and more complex prediction tasks.
