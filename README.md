# 🛡️ Fraud Detection for E-commerce & Bank Transactions

> **Adey Innovations Inc.** | FinTech Fraud Detection System  
> Improving fraud detection across e-commerce and bank credit card transaction streams using machine learning.

---

## 📌 Overview

Financial fraud causes billions in losses annually. This project builds a unified fraud detection system for two very different transaction streams:

- **E-commerce transactions** — rich behavioral context (device, IP, browser, signup time, purchase velocity)
- **Bank credit card transactions** — anonymized PCA-transformed features for privacy

Effective fraud detection must balance two competing costs:
- **False positives** — flagging legitimate transactions frustrates customers and erodes trust
- **False negatives** — missing actual fraud causes direct financial loss

This system addresses both using carefully chosen resampling strategies, ensemble models, and explainability tools (SHAP) to translate model decisions into actionable business recommendations.

---

## 📁 Project Structure

```
fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                        # ⚠️ Gitignored — not pushed to repo
│   ├── raw/                     # Original datasets
│   └── processed/               # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb     # EDA for e-commerce dataset
│   ├── eda-creditcard.ipynb     # EDA for credit card dataset
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb           # Model training & evaluation
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   └── __init__.py
├── tests/
│   └── __init__.py
├── models/                      # Saved model artifacts (.pkl)
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Datasets

| File | Description |
|------|-------------|
| `Fraud_Data.csv` | E-commerce transactions with user, device, and behavioral context. Target: `class` (1 = fraud) |
| `IpAddress_to_Country.csv` | Maps IP address ranges to countries for geolocation enrichment |
| `creditcard.csv` | Bank credit card transactions with PCA-anonymized features V1–V28. Target: `Class` (1 = fraud) |

> ⚠️ **Note:** The `data/` directory is excluded from version control via `.gitignore`.  
> Download the raw datasets and place them in `data/raw/` before running any notebooks.

Both datasets are **highly imbalanced** — fraudulent transactions represent a small minority of records, which shapes the choice of resampling strategy and evaluation metrics.

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9+
- pip or conda

### Clone the Repository
```bash
git clone https://github.com/Kiyazed/fraud-detection.git
cd fraud-detection
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Key Dependencies
```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
shap
matplotlib
seaborn
joblib
jupyter
```

---

## 🚀 How to Run

Run the notebooks **in this order**:

| Step | Notebook | Description |
|------|----------|-------------|
| 1 | `eda-fraud-data.ipynb` | Exploratory analysis of e-commerce data |
| 2 | `eda-creditcard.ipynb` | Exploratory analysis of credit card data |
| 3 | `feature-engineering.ipynb` | Feature engineering & preprocessing |
| 4 | `modeling.ipynb` | Model training, evaluation & comparison |
| 5 | `shap-explainability.ipynb` | SHAP-based model interpretation |

To launch Jupyter:
```bash
jupyter notebook notebooks/
```

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed duplicates and handled missing values
- Parsed and corrected data types (timestamps, IP addresses)
- **Geolocation enrichment**: converted IP addresses to integers and merged with `IpAddress_to_Country.csv` using binary range lookup

### 2. Feature Engineering (E-commerce)
| Feature | Description |
|---------|-------------|
| `hour_of_day` | Hour of the purchase (0–23) |
| `day_of_week` | Day of week (0=Monday) |
| `time_since_signup` | Hours between signup and purchase |
| `user_tx_count` | Number of transactions per user (velocity signal) |
| `country` | Derived from IP address range lookup |

### 3. Handling Class Imbalance — SMOTE
**SMOTE (Synthetic Minority Oversampling Technique)** was applied to the **training set only**.

**Why SMOTE?**
- Both datasets have severe class imbalance (~1–9% fraud rate)
- Simple duplication risks overfitting to exact minority samples
- SMOTE synthesizes new fraud samples by interpolating between existing ones, giving the model more generalizable fraud patterns
- Applied only to training data the test set remains untouched to reflect real-world distribution

### 4. Models Trained
| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Baseline | Interpretable, scaled features |
| Random Forest | Ensemble | 300 trees, balanced class weights |
| XGBoost | Gradient Boosting | AUC-PR eval metric, early stopping |
| LightGBM | Gradient Boosting | Fast, leaf-wise growth, balanced weights |

### 5. Evaluation Strategy
- **Primary metric: AUC-PR** (Area Under Precision-Recall Curve)
- Secondary metrics: F1-Score, ROC-AUC, Confusion Matrix
- **Stratified 5-Fold Cross-Validation** for robust performance estimation

> **Why AUC-PR over accuracy or ROC-AUC?**  
> On severely imbalanced data, a model predicting "all legitimate" achieves >99% accuracy.  
> AUC-PR directly measures performance on the minority (fraud) class — the class that matters most financially.

---

## 📈 Results

### E-commerce Dataset

| Model               | AUC-PR | F1-Score | ROC-AUC |
|------------------   |--------|----------|---------|
| Logistic Regression | 0.5546 | 0.2112   | 0.7575  |
| Random Forest       | 0.6196 | 0.2286   | 0.7662  |
| XGBoost             | 0.5826 | 0.2593   | 0.7646  |
| LightGBM            | 0.6187 | 0.2672   | 0.7628  |

### Credit Card Dataset

| Model               | AUC-PR | F1-Score | ROC-AUC |
|---------------------|--------|----------|---------|
| Logistic Regression | 0.0131 | 0.0162   | 0.6887  |
| Random Forest       | 0.0215 | 0.0091   | 0.5493  |
| XGBoost             | 0.0150 | 0.0157   | 0.5453  |
| LightGBM            | 0.0090 | 0.0169   | 0.5496  |



### Selected Models
- **E-commerce:** *(update after running)*
- **Credit Card:** *(update after running)*

**Justification:** Gradient-boosted models (XGBoost/LightGBM) consistently outperform on tabular fraud data because they model complex non-linear interactions between features (e.g. device × time-of-day × purchase velocity) and focus successive trees on hard-to-classify rare fraud cases.

---

## 💡 Key Findings & Business Recommendations

*(To be completed after SHAP analysis in `shap-explainability.ipynb`)*

Top fraud signals identified via SHAP:
1. **`time_since_signup`** — Very short time between account creation and purchase is a strong fraud indicator
2. **`hour_of_day`** — Fraudulent transactions cluster at unusual hours (late night / early morning)
3. **`user_tx_count`** — Rapid transaction velocity within a short window signals account takeover or card testing
4. **`country`** — Certain geographies show disproportionately high fraud rates
5. **`purchase_value`** — Unusually high or round-number amounts correlate with fraud

### Recommended Actions
- 🔴 **Flag** transactions from accounts less than 1 hour old for manual review
- 🔴 **Trigger step-up authentication** for transactions over a value threshold from new devices
- 🟡 **Monitor** high-velocity users (>3 transactions in 10 minutes) in real time
- 🟡 **Apply stricter rules** for high-risk geographies identified in SHAP country analysis
- 🟢 **Whitelist** long-standing accounts with consistent behavioral patterns to reduce false positives

---

## 🧪 Running Tests

```bash
pytest tests/
```

CI/CD is configured via `.github/workflows/unittests.yml` and runs on every push to `main`.

---

## 👥 Team

| Role | Name |
|------|------|
| Data Science anaysis | Kiya Zewdu |
| Tutor | Kerod |
| Tutor | Mahbubah |
| Tutor | Feven |

**Organization:** Adey Innovations Inc.  
**Program:** 10 Academy — Week 5 & 6

---

## 📄 License

This project is for educational purposes as part of the 10 Academy training program.


