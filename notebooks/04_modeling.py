import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "figure.figsize": (10, 5)})

print("✓ All libraries loaded")

PROJECT_ROOT     = Path("..").resolve()
DATA_DIR         = PROJECT_ROOT / "data"
CLEAN_DATA_PATH  = DATA_DIR / "processed" / "salary_cleaned.csv"
MODEL_DIR        = PROJECT_ROOT / "models"
OUTPUT_DIR       = PROJECT_ROOT / "eda_outputs" / "modeling"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Project Root  :", PROJECT_ROOT)
print("Clean Data    :", CLEAN_DATA_PATH)
print("Model Dir     :", MODEL_DIR)
print("Output Dir    :", OUTPUT_DIR)

assert CLEAN_DATA_PATH.exists(), f"❌ File not found: {CLEAN_DATA_PATH}"
print("\n✓ Cleaned data file found")

df = pd.read_csv(CLEAN_DATA_PATH)

print(f"Shape  : {df.shape[0]:,} rows × {df.shape[1]} columns")
print(f"Targets: Salary From Annual, Salary To Annual")
df.head()

# Encode 'Level' column (contains mixed numeric/alpha like M1, M2, 4B)
le_level = LabelEncoder()
df["Level Encoded"] = le_level.fit_transform(df["Level"].astype(str))

# Final feature columns — all numeric/encoded
FEATURE_COLS = [
    "Career Level Encoded",
    "Job Category Encoded",
    "Agency Encoded",
    "Borough Encoded",
    "Title Seniority Encoded",
    "Title Classification Encoded",
    "Posting Type Encoded",
    "Level Encoded",
    "# Of Positions",
    "Is Full Time",
    "Is External",
]

TARGET_MIN = "Salary From Annual"   # predict minimum salary
TARGET_MAX = "Salary To Annual"     # predict maximum salary

X = df[FEATURE_COLS]
y_min = df[TARGET_MIN]
y_max = df[TARGET_MAX]
Y = df[[TARGET_MIN, TARGET_MAX]]    # both targets together

print(f"Features  : {len(FEATURE_COLS)}")
print(f"Samples   : {len(X):,}")
print(f"Target Min: mean=${y_min.mean():,.0f}, std=${y_min.std():,.0f}")
print(f"Target Max: mean=${y_max.mean():,.0f}, std=${y_max.std():,.0f}")
print()
X.describe().round(1)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

y_train_min, y_test_min = Y_train[TARGET_MIN], Y_test[TARGET_MIN]
y_train_max, y_test_max = Y_train[TARGET_MAX], Y_test[TARGET_MAX]

print(f"Train set : {len(X_train):,} rows ({len(X_train)/len(X)*100:.0f}%)")
print(f"Test set  : {len(X_test):,}  rows ({len(X_test)/len(X)*100:.0f}%)")

def evaluate(y_true, y_pred, label=""):
    """Returns MAE, RMSE, R² for a single target."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    if label:
        print(f"  {label:10s} → MAE: ${mae:>8,.0f}  |  RMSE: ${rmse:>8,.0f}  |  R²: {r2:.4f}")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}

results = {}   # stores all model results for final comparison
print("✓ evaluate() ready")

lr_min = LinearRegression()
lr_max = LinearRegression()

lr_min.fit(X_train, y_train_min)
lr_max.fit(X_train, y_train_max)

pred_min = lr_min.predict(X_test)
pred_max = lr_max.predict(X_test)

print("Linear Regression:")
results["Linear Regression"] = {
    "min": evaluate(y_test_min, pred_min, "Min Salary"),
    "max": evaluate(y_test_max, pred_max, "Max Salary"),
}

ridge_min = Ridge(alpha=10)
ridge_max = Ridge(alpha=10)

ridge_min.fit(X_train, y_train_min)
ridge_max.fit(X_train, y_train_max)

pred_min = ridge_min.predict(X_test)
pred_max = ridge_max.predict(X_test)

print("Ridge Regression (alpha=10):")
results["Ridge"] = {
    "min": evaluate(y_test_min, pred_min, "Min Salary"),
    "max": evaluate(y_test_max, pred_max, "Max Salary"),
}

rf_min = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_max = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

rf_min.fit(X_train, y_train_min)
rf_max.fit(X_train, y_train_max)

pred_min = rf_min.predict(X_test)
pred_max = rf_max.predict(X_test)

print("Random Forest (200 trees):")
results["Random Forest"] = {
    "min": evaluate(y_test_min, pred_min, "Min Salary"),
    "max": evaluate(y_test_max, pred_max, "Max Salary"),
}

xgb_min = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, random_state=42,
                        verbosity=0)
xgb_max = XGBRegressor(n_estimators=300, learning_rate=0.05,
                        max_depth=6, subsample=0.8,
                        colsample_bytree=0.8, random_state=42,
                        verbosity=0)

xgb_min.fit(X_train, y_train_min)
xgb_max.fit(X_train, y_train_max)

pred_min = xgb_min.predict(X_test)
pred_max = xgb_max.predict(X_test)

print("XGBoost (300 estimators):")
results["XGBoost"] = {
    "min": evaluate(y_test_min, pred_min, "Min Salary"),
    "max": evaluate(y_test_max, pred_max, "Max Salary"),
}

lgb_min = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, random_state=42,
                         verbose=-1)
lgb_max = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                         max_depth=6, subsample=0.8,
                         colsample_bytree=0.8, random_state=42,
                         verbose=-1)

lgb_min.fit(X_train, y_train_min)
lgb_max.fit(X_train, y_train_max)

pred_min = lgb_min.predict(X_test)
pred_max = lgb_max.predict(X_test)

print("LightGBM (300 estimators):")
results["LightGBM"] = {
    "min": evaluate(y_test_min, pred_min, "Min Salary"),
    "max": evaluate(y_test_max, pred_max, "Max Salary"),
}

rows = []
for model_name, res in results.items():
    rows.append({
        "Model"        : model_name,
        "MAE (Min)"    : f"${res['min']['MAE']:,.0f}",
        "RMSE (Min)"   : f"${res['min']['RMSE']:,.0f}",
        "R² (Min)"     : f"{res['min']['R2']:.4f}",
        "MAE (Max)"    : f"${res['max']['MAE']:,.0f}",
        "RMSE (Max)"   : f"${res['max']['RMSE']:,.0f}",
        "R² (Max)"     : f"{res['max']['R2']:.4f}",
    })

comparison_df = pd.DataFrame(rows).set_index("Model")
print("\n📊 MODEL COMPARISON")
print("=" * 75)
print(comparison_df.to_string())
print("=" * 75)
comparison_df

model_names = list(results.keys())
mae_min  = [results[m]["min"]["MAE"]  for m in model_names]
mae_max  = [results[m]["max"]["MAE"]  for m in model_names]
r2_min   = [results[m]["min"]["R2"]   for m in model_names]
r2_max   = [results[m]["max"]["R2"]   for m in model_names]

x = np.arange(len(model_names))
w = 0.35

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# MAE comparison
axes[0].bar(x - w/2, mae_min, width=w, color="#378ADD", label="Min Salary")
axes[0].bar(x + w/2, mae_max, width=w, color="#1D9E75", label="Max Salary", alpha=0.85)
axes[0].set_xticks(x)
axes[0].set_xticklabels(model_names, rotation=15, ha="right")
axes[0].set_ylabel("MAE ($)")
axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
axes[0].set_title("Mean Absolute Error (lower = better)", fontsize=12, fontweight="bold")
axes[0].legend()

# R² comparison
axes[1].bar(x - w/2, r2_min, width=w, color="#378ADD", label="Min Salary")
axes[1].bar(x + w/2, r2_max, width=w, color="#1D9E75", label="Max Salary", alpha=0.85)
axes[1].set_xticks(x)
axes[1].set_xticklabels(model_names, rotation=15, ha="right")
axes[1].set_ylabel("R² Score")
axes[1].set_ylim(0, 1)
axes[1].set_title("R² Score (higher = better)", fontsize=12, fontweight="bold")
axes[1].legend()

plt.suptitle("Model Performance Comparison", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_model_comparison.png")
plt.show()
print("✓ Saved:", OUTPUT_DIR / "01_model_comparison.png")

# Rank by average R² across both targets
best_model_name = max(
    results,
    key=lambda m: (results[m]["min"]["R2"] + results[m]["max"]["R2"]) / 2
)
print(f"🏆 Best model: {best_model_name}")
print(f"   R² (Min Salary): {results[best_model_name]['min']['R2']:.4f}")
print(f"   R² (Max Salary): {results[best_model_name]['max']['R2']:.4f}")
print(f"   MAE (Min Salary): ${results[best_model_name]['min']['MAE']:,.0f}")
print(f"   MAE (Max Salary): ${results[best_model_name]['max']['MAE']:,.0f}")

from sklearn.model_selection import GridSearchCV

# Tune XGBoost (adjust if LightGBM wins above)
param_grid = {
    "n_estimators"    : [200, 300, 500],
    "max_depth"       : [4, 6, 8],
    "learning_rate"   : [0.01, 0.05, 0.1],
    "subsample"       : [0.8, 1.0],
}

print("Tuning XGBoost for Min Salary target...")
print("(This may take a few minutes with GridSearchCV)")

xgb_tuned = XGBRegressor(random_state=42, verbosity=0, colsample_bytree=0.8)

grid_search = GridSearchCV(
    xgb_tuned, param_grid,
    cv=5, scoring="neg_mean_absolute_error",
    n_jobs=-1, verbose=1
)
grid_search.fit(X_train, y_train_min)

print(f"\n✓ Best params : {grid_search.best_params_}")
print(f"  Best CV MAE : ${-grid_search.best_score_:,.0f}")

# Retrain best model on both targets with tuned params
best_params = grid_search.best_params_

tuned_min = XGBRegressor(**best_params, random_state=42, verbosity=0, colsample_bytree=0.8)
tuned_max = XGBRegressor(**best_params, random_state=42, verbosity=0, colsample_bytree=0.8)

tuned_min.fit(X_train, y_train_min)
tuned_max.fit(X_train, y_train_max)

pred_min_tuned = tuned_min.predict(X_test)
pred_max_tuned = tuned_max.predict(X_test)

print("Tuned XGBoost:")
results["XGBoost (Tuned)"] = {
    "min": evaluate(y_test_min, pred_min_tuned, "Min Salary"),
    "max": evaluate(y_test_max, pred_max_tuned, "Max Salary"),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_min = cross_val_score(tuned_min, X, y_min,
                          cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)
cv_max = cross_val_score(tuned_max, X, y_max,
                          cv=kf, scoring="neg_mean_absolute_error", n_jobs=-1)

print("5-Fold Cross-Validation (Tuned XGBoost):")
print(f"  Min Salary → MAE per fold: {[-round(v,0) for v in cv_min]}")
print(f"  Min Salary → Mean MAE: ${-cv_min.mean():,.0f}  ±  ${cv_min.std():,.0f}")
print()
print(f"  Max Salary → MAE per fold: {[-round(v,0) for v in cv_max]}")
print(f"  Max Salary → Mean MAE: ${-cv_max.mean():,.0f}  ±  ${cv_max.std():,.0f}")

feat_imp_min = pd.Series(tuned_min.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
feat_imp_max = pd.Series(tuned_max.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

feat_imp_min.plot(kind="barh", ax=axes[0], color="#378ADD")
axes[0].set_title("Feature Importance — Min Salary", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Importance Score")

feat_imp_max.plot(kind="barh", ax=axes[1], color="#1D9E75")
axes[1].set_title("Feature Importance — Max Salary", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Importance Score")

plt.suptitle("XGBoost Feature Importances", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_feature_importance.png")
plt.show()
print("✓ Saved:", OUTPUT_DIR / "02_feature_importance.png")

residuals_min = y_test_min.values - pred_min_tuned
residuals_max = y_test_max.values - pred_max_tuned

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Residual scatter — Min
axes[0,0].scatter(pred_min_tuned, residuals_min, alpha=0.3, s=12, color="#378ADD")
axes[0,0].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0,0].set_xlabel("Predicted Min Salary")
axes[0,0].set_ylabel("Residual ($)")
axes[0,0].set_title("Residuals vs Predicted — Min Salary", fontweight="bold")
axes[0,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
axes[0,0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

# Residual scatter — Max
axes[0,1].scatter(pred_max_tuned, residuals_max, alpha=0.3, s=12, color="#1D9E75")
axes[0,1].axhline(0, color="red", linestyle="--", linewidth=1)
axes[0,1].set_xlabel("Predicted Max Salary")
axes[0,1].set_ylabel("Residual ($)")
axes[0,1].set_title("Residuals vs Predicted — Max Salary", fontweight="bold")
axes[0,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
axes[0,1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

# Residual histogram — Min
axes[1,0].hist(residuals_min, bins=40, color="#378ADD", edgecolor="white")
axes[1,0].axvline(0, color="red", linestyle="--")
axes[1,0].set_xlabel("Residual ($)")
axes[1,0].set_title("Residual Distribution — Min Salary", fontweight="bold")
axes[1,0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

# Residual histogram — Max
axes[1,1].hist(residuals_max, bins=40, color="#1D9E75", edgecolor="white")
axes[1,1].axvline(0, color="red", linestyle="--")
axes[1,1].set_xlabel("Residual ($)")
axes[1,1].set_title("Residual Distribution — Max Salary", fontweight="bold")
axes[1,1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))

plt.suptitle("Residual Analysis — Tuned XGBoost", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "03_residual_analysis.png")
plt.show()
print("✓ Saved:", OUTPUT_DIR / "03_residual_analysis.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, y_true, y_pred, color, label in zip(
    axes,
    [y_test_min, y_test_max],
    [pred_min_tuned, pred_max_tuned],
    ["#378ADD", "#1D9E75"],
    ["Min Salary", "Max Salary"]
):
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color=color)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax.set_xlabel(f"Actual {label} ($)")
    ax.set_ylabel(f"Predicted {label} ($)")
    ax.set_title(f"Actual vs Predicted — {label}", fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v/1000:.0f}k"))
    ax.legend(fontsize=9)

plt.suptitle("Actual vs Predicted Salary — Tuned XGBoost", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "04_actual_vs_predicted.png")
plt.show()
print("✓ Saved:", OUTPUT_DIR / "04_actual_vs_predicted.png")

violations = np.sum(pred_min_tuned > pred_max_tuned)
total = len(pred_min_tuned)

print(f"Predictions where Min > Max : {violations} / {total}")
print(f"Violation rate              : {violations/total*100:.2f}%")

if violations > 0:
    print("\n⚠️  Applying post-processing fix: swap predictions where min > max")
    fixed_min = np.where(pred_min_tuned > pred_max_tuned, pred_max_tuned, pred_min_tuned)
    fixed_max = np.where(pred_min_tuned > pred_max_tuned, pred_min_tuned, pred_max_tuned)
    print("✓ Fix applied")
else:
    print("\n✓ No violations — all predictions satisfy Min ≤ Max")
    fixed_min = pred_min_tuned
    fixed_max = pred_max_tuned

print("=" * 65)
print("  FINAL MODEL RESULTS — TUNED XGBoost")
print("=" * 65)
print()
print("  MIN SALARY (Salary From Annual):")
min_res = evaluate(y_test_min, fixed_min)
print(f"    MAE  : ${min_res['MAE']:>10,.0f}   (avg prediction error)")
print(f"    RMSE : ${min_res['RMSE']:>10,.0f}")
print(f"    R²   : {min_res['R2']:>10.4f}   (1.0 = perfect)")
print()
print("  MAX SALARY (Salary To Annual):")
max_res = evaluate(y_test_max, fixed_max)
print(f"    MAE  : ${max_res['MAE']:>10,.0f}")
print(f"    RMSE : ${max_res['RMSE']:>10,.0f}")
print(f"    R²   : {max_res['R2']:>10.4f}")
print()
print(f"  Min ≤ Max violations : {np.sum(fixed_min > fixed_max)} / {len(fixed_min)}")
print("=" * 65)

joblib.dump(tuned_min, MODEL_DIR / "xgb_salary_min.pkl")
joblib.dump(tuned_max, MODEL_DIR / "xgb_salary_max.pkl")

print("✓ Models saved:")
print(f"   {MODEL_DIR / 'xgb_salary_min.pkl'}")
print(f"   {MODEL_DIR / 'xgb_salary_max.pkl'}")

def predict_salary(career_level_encoded, job_category_encoded,
                   agency_encoded, borough_encoded,
                   title_seniority_encoded, title_classification_encoded,
                   posting_type_encoded, level_encoded,
                   num_positions=1, is_full_time=1, is_external=1):
    """
    Predict min and max salary for a job posting.

    All *_encoded params are the integer-encoded values from preprocessing.
    Returns: (predicted_min, predicted_max)
    """
    model_min = joblib.load(MODEL_DIR / "xgb_salary_min.pkl")
    model_max = joblib.load(MODEL_DIR / "xgb_salary_max.pkl")

    features = pd.DataFrame([{
        "Career Level Encoded"         : career_level_encoded,
        "Job Category Encoded"         : job_category_encoded,
        "Agency Encoded"               : agency_encoded,
        "Borough Encoded"              : borough_encoded,
        "Title Seniority Encoded"      : title_seniority_encoded,
        "Title Classification Encoded" : title_classification_encoded,
        "Posting Type Encoded"         : posting_type_encoded,
        "Level Encoded"                : level_encoded,
        "# Of Positions"               : num_positions,
        "Is Full Time"                 : is_full_time,
        "Is External"                  : is_external,
    }])

    pred_min = model_min.predict(features)[0]
    pred_max = model_max.predict(features)[0]

    # Enforce constraint: min <= max
    if pred_min > pred_max:
        pred_min, pred_max = pred_max, pred_min

    return round(pred_min, 2), round(pred_max, 2)


# ── Example prediction ────────────────────────────────────────
# Use encoded values from your LabelEncoder mappings
sample = X_test.iloc[0]
pred_min_ex, pred_max_ex = predict_salary(
    career_level_encoded         = int(sample["Career Level Encoded"]),
    job_category_encoded         = int(sample["Job Category Encoded"]),
    agency_encoded               = int(sample["Agency Encoded"]),
    borough_encoded              = int(sample["Borough Encoded"]),
    title_seniority_encoded      = int(sample["Title Seniority Encoded"]),
    title_classification_encoded = int(sample["Title Classification Encoded"]),
    posting_type_encoded         = int(sample["Posting Type Encoded"]),
    level_encoded                = int(sample["Level Encoded"]),
    num_positions                = int(sample["# Of Positions"]),
    is_full_time                 = int(sample["Is Full Time"]),
    is_external                  = int(sample["Is External"]),
)

actual_min = y_test_min.iloc[0]
actual_max = y_test_max.iloc[0]

print(f"Example Prediction:")
print(f"  Predicted : ${pred_min_ex:,.0f}  –  ${pred_max_ex:,.0f}")
print(f"  Actual    : ${actual_min:,.0f}  –  ${actual_max:,.0f}")
print(f"  Error     : ${abs(pred_min_ex - actual_min):,.0f} (min)  |  ${abs(pred_max_ex - actual_max):,.0f} (max)")
