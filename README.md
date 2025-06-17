# Project: E-Commerce Shipments Delivery Prediction

This project focuses on predicting whether product deliveries for an international electronics retailer will arrive **on time or be delayed**, using historical order data. The solution leverages machine learning algorithms such as **Logistic Regression, Random Forest, XGBoost, and LightGBM** for classification.

The entire workflow is tracked using **MLflow** for experiment management, and the final model is deployed through a **Streamlit dashboard** to provide an interactive interface for business users and logistics planners.

---

##  Why This Project?

- **Business relevance**: Delivery performance is crucial to customer satisfaction and logistics cost management.
- **Classification problem**: A well-known binary classification use case with real-world applications.
- **Skill showcase**: Includes preprocessing, model selection, hyperparameter tuning, MLflow tracking, and Streamlit deployment.
- **Explainability**: Uses interpretable features and includes model evaluation comparison across 4 algorithms.

---

##  Dataset

- **Source**: [Kaggle – E-Commerce Shipping Data](https://www.kaggle.com/datasets/prachi13/customer-analytics)
- **Size**: ~11,000 rows  
- **Target column**: `Reached.on.Time_Y.N` (1 = On time, 0 = Delayed)

**Key Features:**
- `Customer_rating`: Rating (1 to 5)
- `Cost_of_the_Product`
- `Discount_offered`
- `Weight_in_gms`
- `Mode_of_Shipment` (Categorical)
- `Product_importance` (Categorical)
- `Cost_per_gm`: Derived feature

---

##  Project Workflow

### 1. **Data Cleaning & Preprocessing**
- Removed duplicate and missing entries
- Feature engineering (`Cost_per_gm`)
- Categorical encoding (One-hot)

### 2. **Exploratory Data Analysis (EDA)**
- Distribution plots of numerical/categorical features
- Correlation heatmaps
- Class imbalance check

### 3. **Model Building**
- Trained the following models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
- Performed hyperparameter tuning for each model

### 4. **Evaluation Metrics**
- Accuracy, Precision, Recall, AUC-ROC
- Used `classification_report`, `confusion_matrix`, `roc_auc_score`

### 5. **Model Selection**
- Chose XGBoost for deployment due to highest **Recall** (0.747) and **AUC** (0.752)

### 6. **MLflow Tracking**
- Logged model parameters, metrics, artifacts

### 7. **Streamlit Deployment**
- Built an interactive app with:
  - Input form for predictions
  - Results and model confidence
  - Streamlit Web App: A user-friendly Streamlit app has been deployed to interactively explore the model and make shipment predictions.

### You can access the live app here: https://wpa6ubwgqccj3aguc22rvr.streamlit.app/
---

##  Model Evaluation Summary

### Insights & Interpretation

- **XGBoost** achieved the highest recall (0.747) and highest AUC (0.752), making it best for minimizing false negatives.
- **LightGBM** had the highest precision (0.860) but a low recall (0.543).
- **Random Forest** had very high precision (0.868) but lower recall (0.534).
- **Logistic Regression** was balanced with precision 0.733 and recall 0.692.

### Conclusion

- **Use XGBoost** when detecting delays is the top priority.
- **Use LightGBM** when false positives are more expensive.
- Other models can serve as interpretable baselines or backups.

### Model Comparison Table

| Model               | Accuracy | Precision | Recall | AUC   | Comments                                              |
|---------------------|----------|-----------|--------|-------|--------------------------------------------------------|
| Logistic Regression | 0.666    | 0.733     | 0.692  | 0.737 | Balanced metrics; interpretable and consistent         |
| Random Forest       | 0.673    | 0.868     | 0.534  | 0.742 | Very high precision; weaker recall                     |
| XGBoost             | 0.656    | 0.698     | 0.747  | 0.752 | Best recall and AUC; strong for capturing positives    |
| LightGBM            | 0.675    | 0.860     | 0.543  | 0.744 | Highest precision; misses many positives (low recall)  |

---

##  Project Structure

```
E-Commerce_Prediction_Project/
│
├── data/
│   └── shipping_data.csv
├── models/
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── train_columns.pkl
├── mlruns/
│   └── ... (MLflow tracking logs)
├── app.py
├── notebooks/project.ipynb
├── requirements.txt
└── README.md
```
---

##  Run the App Locally

```bash
# Clone the repository
git clone https://github.com/JyotiTidke26/E-Commerce_Prediction_Project.git
cd E-Commerce_Prediction_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

##  requirements.txt

```txt
streamlit
scikit-learn
xgboost
lightgbm
pandas
numpy
joblib
matplotlib
seaborn
mlflow
```
---

## Future Improvements

- Apply SMOTE or other techniques for class balancing to improve model training on imbalanced data.
- Integrate SHAP (SHapley Additive exPlanations) for enhanced model interpretability and explainability.
- Enable storing and downloading batch prediction results for user convenience.
- Deploy the application using Docker containers or on cloud platforms like AWS, GCP, or Azure for scalability.
- Add a batch CSV upload feature in the Streamlit app for bulk predictions.

---
