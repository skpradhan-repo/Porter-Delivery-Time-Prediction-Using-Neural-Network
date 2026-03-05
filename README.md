# Porter-Delivery-Time-Prediction-Using-Neural-Network
Predicting intra-city delivery times for Porter using machine learning and neural networks. Includes data preprocessing, exploratory analysis, outlier treatment, Random Forest baseline, and neural network regression with feature scaling and evaluation metrics.

---

## 📌 Project Overview

Porter is India’s largest marketplace for intra-city logistics, servicing millions of customers with thousands of deliveries daily. Accurate delivery time prediction is crucial for:

- Providing reliable ETAs to customers
- Efficiently allocating dashers
- Improving operational efficiency and customer satisfaction

This project uses historical delivery data to build predictive models for delivery time estimation based on features like order details, restaurant category, time of day, and available dashers.

---

## 🗂 Dataset

Each row represents a unique delivery. Key features include:

- `market_id` – ID of the city/market  
- `store_primary_category` – Restaurant category  
- `order_protocol` – How the order was placed (online, call, third-party)  
- `total_items`, `subtotal`, `num_distinct_items`, `min_item_price`, `max_item_price` – Order details  
- `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders` – Dasher availability  
- `estimated_store_to_consumer_driving_duration` – Approximate travel time  
- `created_at`, `actual_delivery_time` – Timestamps for order and delivery  
- `Time_taken_for_delivery` – Target variable (in minutes)  

---

## 🧹 Data Preprocessing

- Converted timestamps to datetime objects
- Created target variable: `Time_taken_for_delivery`
- Extracted additional features: `hour`, `day`, `month`
- Encoded categorical variables using label encoding
- Handled outliers using **1st and 99th percentile capping**
- Scaled features with **MinMaxScaler** for neural network training

---

## 📊 Exploratory Data Analysis (EDA)

- Correlation analysis to understand feature importance
- Visualizations including:
  - Histograms & boxplots for continuous features
  - Countplots for categorical features (`day`, `hour`, `order_protocol`)
  - Scatterplots to analyze relationships with delivery time
- Identified key drivers of delivery time:
  - `estimated_store_to_consumer_driving_duration`
  - `subtotal`, `num_distinct_items`
  - Dashers on shift and busy dashers

---

## 🤖 Modeling

### Random Forest Regressor
- Trained as a baseline model
- Metrics:
  - **R² ≈ 0.94**
  - **RMSE ≈ 1.83 minutes**
  - Good predictive power, but limited in capturing complex non-linear relationships

### Neural Network Regression
- Feedforward network with 3 hidden layers (64 → 32 → 16 neurons)
- ReLU activations for hidden layers, Linear activation for output
- BatchNormalization + Dropout to stabilize training
- Adam optimizer for efficient gradient updates
- Metrics:
  - **R² ≈ 0.959**
  - **RMSE ≈ 1.55 minutes**
  - **MAPE ≈ 2.6%** — highly accurate predictions

---

## 📈 Key Business Insights

1. **Reliable ETAs for Customers:**  
   - Average prediction error ≈ 1.5 minutes, allowing precise delivery estimates.

2. **Efficient Resource Allocation:**  
   - Dashers can be scheduled optimally based on predicted delivery times.

3. **Operational Planning:**  
   - High-impact features like order size, number of items, and distance can be monitored for improving delivery efficiency.

4. **Scalability:**  
   - Neural network performs well on large datasets, enabling expansion to new markets and larger order volumes.

---

## 🔧 Future Improvements

- Include traffic, weather, and peak-hour data for more accurate predictions
- Hyperparameter tuning and cross-validation for neural network optimization
- Explore advanced architectures (e.g., LSTM) for sequential order patterns
- Build a real-time ETA prediction API for Porter’s operations

---

## 📁 Repository Structure
Porter-Delivery-Time-Prediction-Using-Neural-Network/
│
├── data/
│   └── data_2.csv              (or rename to porter_delivery_data.csv)
│
├── notebooks/
│   └── Porter_Neural_Network.ipynb     # Main + only notebook (EDA → modeling → evaluation)
│
├── models/
│   └── porter_delivery_time_model.h5
│
├── reports/
│   └── figures/                # Save your important plots here
│       ├── loss_curve.png
│       ├── delivery_time_vs_hour.png
│       └── boxplot.png
│
└── README.md

---
## ⚙️ Requirements

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
tensorflow
statsmodels
