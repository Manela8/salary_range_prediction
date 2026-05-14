# 💼 Salary Range Prediction — NYC Job Postings

> A machine learning project to predict **minimum and maximum salary ranges** for NYC government job postings based on job attributes such as title, category, career level, and location.

---

## 📁 Project Structure

```
salary_range_prediction/
│
├── data/
│   ├── raw/
│   │   └── Jobs_NYC_Postings_-_Jobs_NYC_Postings.csv   ← original dataset (do not modify)
│   └── processed/
│       └── salary_cleaned.csv                          ← cleaned & feature-engineered data
│
├── notebooks/
│   ├── 01_eda_raw.ipynb          ← Phase 1: EDA on raw data
│   ├── 02_preprocessing.ipynb   ← Phase 2: Cleaning & feature engineering
│   ├── 03_eda_cleaned.ipynb      ← Phase 3: EDA on cleaned data
│   └── 04_modeling.ipynb         ← Phase 4: Model training & evaluation
│
├── models/
│   ├── xgb_salary_min.pkl        ← trained model for minimum salary
│   └── xgb_salary_max.pkl        ← trained model for maximum salary
│
├── eda_outputs/
│   ├── raw/                      ← charts from Phase 1 EDA
│   ├── cleaned/                  ← charts from Phase 3 EDA
│   └── modeling/                 ← charts from Phase 4 modeling
│
├── requirements.txt              ← all Python dependencies
└── README.md                     ← you are here
```

---

## 🎯 Project Objective

Develop a predictive model that estimates **salary ranges** (minimum and maximum) for job listings using historical NYC job posting data. This helps:

- Recruiters align postings with market standards
- Candidates get transparent salary expectations upfront
- HR leaders make informed budget decisions

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | NYC Open Data — Job Postings |
| Raw Records | 5,120 rows |
| Columns | 30 |
| After Cleaning | 5,074 rows |
| Salary Types | Annual · Hourly · Daily |

**Prediction Targets:**
- `Salary From Annual` — minimum salary (annualized)
- `Salary To Annual` — maximum salary (annualized)

---

## 🔧 How to Run

### 1. Clone / download the project
```bash
git clone <your-repo-url>
cd salary_range_prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run notebooks in order
```bash
cd notebooks

# Option A — Run as Jupyter notebooks
jupyter notebook

# Option B — Convert to .py and run from terminal
jupyter nbconvert --to script 04_modeling.ipynb --output 04_modeling.py
python 04_modeling.py
```

> ⚠️ Always run in order: `01` → `02` → `03` → `04`

---

## 🤖 Models Trained

| Model | Description |
|---|---|
| Linear Regression | Baseline model |
| Ridge Regression | L2 regularized baseline |
| Random Forest | 200 decision trees |
| XGBoost | Gradient boosting (300 estimators) |
| LightGBM | Fast gradient boosting (300 estimators) |

**Best model:** XGBoost (tuned with GridSearchCV, 5-fold cross-validation)

---

## 📐 Features Used

| Feature | Description |
|---|---|
| `Career Level Encoded` | Ordinal: Entry(1) → Executive(5) |
| `Job Category Encoded` | Label encoded job category |
| `Agency Encoded` | Label encoded hiring agency |
| `Borough Encoded` | Extracted & encoded from Work Location |
| `Title Seniority Encoded` | Extracted from Business Title (Junior/Senior/Manager etc.) |
| `Title Classification Encoded` | Label encoded title classification |
| `Posting Type Encoded` | Internal vs External posting |
| `Level Encoded` | Civil service level (encoded) |
| `# Of Positions` | Number of open positions |
| `Is Full Time` | Binary flag (1 = Full Time) |
| `Is External` | Binary flag (1 = External posting) |

---

## 📈 Evaluation Metrics

| Metric | Description |
|---|---|
| **MAE** | Mean Absolute Error — average dollar error |
| **RMSE** | Root Mean Squared Error — penalizes large errors |
| **R²** | Coefficient of determination — 1.0 = perfect fit |

---

## 🔮 Making Predictions

After running `04_modeling.ipynb`, use the saved models:

```python
import joblib
import pandas as pd

model_min = joblib.load("models/xgb_salary_min.pkl")
model_max = joblib.load("models/xgb_salary_max.pkl")

features = pd.DataFrame([{
    "Career Level Encoded"         : 3,   # Experienced
    "Job Category Encoded"         : 5,   # e.g. Technology
    "Agency Encoded"               : 10,
    "Borough Encoded"              : 2,   # Manhattan
    "Title Seniority Encoded"      : 4,   # Senior
    "Title Classification Encoded" : 1,
    "Posting Type Encoded"         : 1,   # External
    "Level Encoded"                : 2,
    "# Of Positions"               : 1,
    "Is Full Time"                 : 1,
    "Is External"                  : 1,
}])

pred_min = model_min.predict(features)[0]
pred_max = model_max.predict(features)[0]

print(f"Predicted Salary Range: ${pred_min:,.0f} — ${pred_max:,.0f}")
```

---

## 📦 Dependencies

See [`requirements.txt`](requirements.txt) for the full list.

Key libraries:
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — ML models & evaluation
- `xgboost`, `lightgbm` — gradient boosting
- `joblib` — model saving/loading

---

## 👤 Author

**Project:** Salary Range Prediction  
**Dataset:** NYC Open Data — Job Postings  
**Tools:** Python · Jupyter · scikit-learn · XGBoost · LightGBM
