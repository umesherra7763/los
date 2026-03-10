"""
=============================================================
STEP 3: DATA PREPROCESSING PIPELINE
=============================================================
Main Dataset: synthetic_hospital_data.csv (5000 × 44)

Preprocessing Steps:
  1. Drop irrelevant / leakage columns
  2. Encode categorical variables
     - Label Encoding  → binary / ordinal categories
     - One-Hot Encoding → nominal categories
  3. Feature Scaling
     - StandardScaler  → continuous numeric features
  4. Feature Selection
     - Correlation analysis (remove redundant features)
     - Chi-squared test (for categorical vs target)
  5. Train / Test Split (80% / 20%)
  6. Save all outputs

Why each step matters is explained inline.
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression, chi2
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("  STEP 3: DATA PREPROCESSING PIPELINE")
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(SCRIPT_DIR, 'synthetic_hospital_data.csv'))
print(f"\n✅ Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────
# STEP 1: DROP IRRELEVANT / LEAKAGE COLUMNS
# ─────────────────────────────────────────────────────────────
"""
WHAT IS COLUMN LEAKAGE?
Leakage = using information in features that would NOT be
available at prediction time (when patient is admitted).

Example:
  discharge_date     → only known AFTER the patient leaves
  los_category       → derived FROM length_of_stay (our target)
  prolonged_los_flag → derived FROM length_of_stay (our target)
  readmission_30d    → only known 30 days AFTER discharge

These must be DROPPED or the model will cheat.

Also drop:
  patient_id         → just an identifier, no predictive value
  admission_date     → date string, already encoded as month/quarter/year
  discharge_date     → leakage
  day_of_admission   → already captured in weekend_admission flag
"""

drop_cols = [
    'patient_id',           # identifier
    'admission_date',       # string date - encoded as month/year/quarter
    'discharge_date',       # LEAKAGE: only known after discharge
    'los_category',         # LEAKAGE: derived from target
    'prolonged_los_flag',   # LEAKAGE: derived from target
    'readmission_30d',      # LEAKAGE: only known post-discharge
    'day_of_admission',     # redundant with weekend_admission
]

df_clean = df.drop(columns=drop_cols)
print(f"\n[1/5] Dropped {len(drop_cols)} columns (leakage + irrelevant)")
print(f"      Remaining: {df_clean.shape[1]} columns")
print(f"      Dropped: {drop_cols}")

# ─────────────────────────────────────────────────────────────
# STEP 2: ENCODE CATEGORICAL VARIABLES
# ─────────────────────────────────────────────────────────────
"""
ML models only understand NUMBERS, not strings like "Male" or "Emergency".
We convert categories to numbers using two strategies:

LABEL ENCODING → when category has a natural ORDER (ordinal)
  Example: severity_of_illness: Low=1, Medium=2, High=3
  Or binary: Male=0, Female=1

ONE-HOT ENCODING → when category has NO order (nominal)
  Example: department → create a separate 0/1 column for each dept
  Why not label encode here? Because Cardiology=1, Neurology=2
  would falsely imply Neurology > Cardiology, which is meaningless.

Source: Standard ML preprocessing practice (Journal 4 used
        one-hot encoding for categorical features).
"""

print(f"\n[2/5] Encoding categorical variables...")

# ── Label Encoding (binary / ordinal) ───────────────────────
label_encode_cols = {
    'gender':               {'Male': 0, 'Female': 1},
    'admission_type':       {'Elective': 0, 'Emergency': 1},
    'socioeconomic_status': {'Low': 0, 'Middle': 1, 'High': 2},
    'insurance_type':       {'Uninsured': 0, 'Government': 1, 'Private': 2},
    'marital_status':       {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3},
    'hour_of_admission':    {
        '09:00-12:00': 0, '12:00-15:00': 1,
        '15:00-18:00': 2, '18:00-21:00': 3
    },
    'admission_quarter':    {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4},
    'admission_month': {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    },
}

for col, mapping in label_encode_cols.items():
    df_clean[col] = df_clean[col].map(mapping)
    print(f"      Label encoded: {col} → {mapping}")

# ── One-Hot Encoding (nominal) ───────────────────────────────
ohe_cols = ['department', 'primary_diagnosis', 'surgery_type',
            'discharge_disposition']

df_clean = pd.get_dummies(df_clean, columns=ohe_cols, drop_first=True)
print(f"\n      One-hot encoded: {ohe_cols}")
print(f"      Shape after encoding: {df_clean.shape}")

# ─────────────────────────────────────────────────────────────
# STEP 3: FEATURE SCALING
# ─────────────────────────────────────────────────────────────
"""
WHY SCALE?
Many ML models (especially distance-based ones) are sensitive
to feature magnitude. Without scaling:
  - blood_glucose (60–400) dominates over
  - bmi (15–50) just because its numbers are bigger

StandardScaler: transforms each feature to mean=0, std=1
Formula: z = (x - mean) / std

We only scale CONTINUOUS features — NOT binary (0/1) flags
because those are already on the same scale.

Note: Tree-based models (Random Forest, XGBoost) don't NEED
scaling but it doesn't hurt and we scale for completeness
since we may also run Linear Regression and Neural Networks.
"""

print(f"\n[3/5] Scaling continuous features (StandardScaler)...")

# Identify continuous numeric columns to scale
# (exclude binary flags and the target variable)
binary_cols = [c for c in df_clean.columns if df_clean[c].nunique() == 2]
target_col  = 'length_of_stay'

# Columns to scale: numeric, not binary, not target
scale_cols = [
    c for c in df_clean.select_dtypes(include=[np.number]).columns
    if c not in binary_cols and c != target_col
]

print(f"      Scaling {len(scale_cols)} continuous features:")
print(f"      {scale_cols}")

scaler = StandardScaler()
df_scaled = df_clean.copy()
df_scaled[scale_cols] = scaler.fit_transform(df_clean[scale_cols])

print(f"\n      ✅ Scaling complete")
print(f"      Sample — age before: mean={df_clean['age'].mean():.1f}, std={df_clean['age'].std():.1f}")
print(f"      Sample — age after:  mean={df_scaled['age'].mean():.2f}, std={df_scaled['age'].std():.2f}")

# ─────────────────────────────────────────────────────────────
# STEP 4: FEATURE SELECTION
# ─────────────────────────────────────────────────────────────
"""
WHY SELECT FEATURES?
After one-hot encoding we have many columns. Not all are useful.
Too many features can cause:
  - Overfitting (model memorizes noise)
  - Slower training
  - Reduced interpretability

Methods we use:

1. CORRELATION ANALYSIS
   - If two features are highly correlated (r > 0.85),
     they carry the same information — drop one
   - Example: outpatient_visits_30d and outpatient_visits_180d
     are naturally correlated (more short-term = more long-term)

2. SelectKBest with f_regression
   - Scores each feature by its statistical relationship with LOS
   - Keeps top K features
   - Based on F-statistic: how much variance in LOS does this
     feature explain?
   - Journal 4 used Lasso regularization similarly
   - Journal 5 used chi-squared SelectKBest (same concept)
"""

print(f"\n[4/5] Feature Selection...")

# Separate features and target
X = df_scaled.drop(columns=[target_col])
y = df_scaled[target_col]

# ── 4a: Remove highly correlated features ───────────────────
print(f"\n      4a. Correlation analysis (threshold = 0.85)...")
corr_matrix = X.select_dtypes(include=[np.number]).corr().abs()
upper_tri = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)
high_corr_cols = [col for col in upper_tri.columns
                  if any(upper_tri[col] > 0.85)]

if high_corr_cols:
    print(f"      Dropping {len(high_corr_cols)} highly correlated features:")
    for c in high_corr_cols:
        corr_with = upper_tri[c][upper_tri[c] > 0.85].index.tolist()
        print(f"        - {c} (correlated with {corr_with})")
    X = X.drop(columns=high_corr_cols)
else:
    print(f"      No highly correlated features found (all r < 0.85)")

print(f"      Features remaining: {X.shape[1]}")

# ── 4b: SelectKBest — top 30 features ───────────────────────
print(f"\n      4b. SelectKBest (f_regression, top 30 features)...")

# Only use numeric columns for SelectKBest
X_numeric = X.select_dtypes(include=[np.number])

selector = SelectKBest(score_func=f_regression, k=min(30, X_numeric.shape[1]))
selector.fit(X_numeric, y)

# Get selected feature names and scores
feature_scores = pd.DataFrame({
    'feature': X_numeric.columns,
    'f_score': selector.scores_,
    'p_value': selector.pvalues_
}).sort_values('f_score', ascending=False)

selected_features = X_numeric.columns[selector.get_support()].tolist()

print(f"\n      Top 15 Features by F-Score (relationship with LOS):")
print(f"      {'Feature':<40} {'F-Score':>10}  {'p-value':>12}")
print(f"      {'-'*65}")
for _, row in feature_scores.head(15).iterrows():
    sig = "✅" if row['p_value'] < 0.05 else "❌"
    print(f"      {sig} {row['feature']:<38} {row['f_score']:>10.2f}  {row['p_value']:>12.6f}")

# Final feature set
X_final = X_numeric[selected_features]
print(f"\n      ✅ Final feature set: {X_final.shape[1]} features selected")

# ─────────────────────────────────────────────────────────────
# STEP 5: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
"""
WHY SPLIT?
We need to evaluate how well our model generalises to
UNSEEN data. If we train and test on the same data,
the model just memorises — this is called overfitting.

80% Train → model learns from this
20% Test  → model is evaluated on this (never seen during training)

random_state=42 → ensures reproducibility (same split every run)
stratify        → we don't stratify for regression (LOS is continuous)

Journal 4 used 70/30 split. Journal 5 used 80/20.
We use 80/20 as it's more common for datasets of this size.
"""

print(f"\n[5/5] Train / Test Split (80% train, 20% test)...")

X_train, X_test, y_train, y_test = train_test_split(
    X_final, y,
    test_size=0.20,
    random_state=42
)

print(f"\n      Training set   : {X_train.shape[0]} samples × {X_train.shape[1]} features")
print(f"      Test set       : {X_test.shape[0]} samples × {X_test.shape[1]} features")
print(f"      Target (train) : mean={y_train.mean():.2f} days, std={y_train.std():.2f}")
print(f"      Target (test)  : mean={y_test.mean():.2f} days, std={y_test.std():.2f}")

# ─────────────────────────────────────────────────────────────
# SAVE ALL OUTPUTS
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  SAVING OUTPUTS")
print(f"{'='*60}")

base = SCRIPT_DIR + os.sep

# Save train and test sets
X_train.to_csv(base + 'X_train.csv', index=False)
X_test.to_csv(base + 'X_test.csv', index=False)
y_train.to_csv(base + 'y_train.csv', index=False)
y_test.to_csv(base + 'y_test.csv', index=False)

# Save full preprocessed dataset (all features, before SelectKBest)
df_full_preprocessed = pd.concat([X_numeric, y], axis=1)
df_full_preprocessed.to_csv(base + 'preprocessed_full.csv', index=False)

# Save feature importance scores
feature_scores.to_csv(base + 'feature_scores.csv', index=False)

print(f"\n  ✅ X_train.csv          → {X_train.shape}")
print(f"  ✅ X_test.csv           → {X_test.shape}")
print(f"  ✅ y_train.csv          → {y_train.shape}")
print(f"  ✅ y_test.csv           → {y_test.shape}")
print(f"  ✅ preprocessed_full.csv → {df_full_preprocessed.shape}")
print(f"  ✅ feature_scores.csv   → feature rankings saved")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  PREPROCESSING SUMMARY")
print(f"{'='*60}")
print(f"""
  Original dataset      : 5,000 rows × 44 columns
  After dropping leakage: 5,000 rows × 37 columns
  After encoding        : 5,000 rows × {df_scaled.shape[1]} columns
  After scaling         : 5,000 rows × {df_scaled.shape[1]} columns (values changed)
  After corr. filter    : 5,000 rows × {X.shape[1]} features
  After SelectKBest     : 5,000 rows × {X_final.shape[1]} features
  ─────────────────────────────────────
  Final Training set    : {X_train.shape[0]} rows × {X_train.shape[1]} features
  Final Test set        : {X_test.shape[0]} rows × {X_test.shape[1]} features
  Target variable       : length_of_stay (continuous, regression)
""")
print(f"  ✅ PREPROCESSING COMPLETE — Ready for ML models!")
print(f"{'='*60}")
