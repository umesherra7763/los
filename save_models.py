"""
Save trained ML models + feature info into model_bundle.pkl
Run this AFTER preprocessing.py
"""
import os, pickle
import pandas as pd
import numpy as np
from sklearn.linear_model    import LinearRegression, Ridge
from sklearn.tree            import DecisionTreeRegressor
from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing   import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
base       = SCRIPT_DIR + os.sep

print("=" * 50)
print("  SAVE MODELS")
print("=" * 50)

# ── Load preprocessed data ───────────────────────────
X_train = pd.read_csv(base + 'X_train.csv')
X_test  = pd.read_csv(base + 'X_test.csv')
y_train = pd.read_csv(base + 'y_train.csv').squeeze()
y_test  = pd.read_csv(base + 'y_test.csv').squeeze()

print(f"\n✅ Data loaded: {X_train.shape[0]} train, {X_test.shape[0]} test samples")

# ── Identify continuous cols for scaling ─────────────
binary_cols = [c for c in X_train.columns if X_train[c].nunique() == 2]
continuous_cols = [c for c in X_train.select_dtypes(include=[np.number]).columns
                   if c not in binary_cols]

# ── Fit scaler on training data ──────────────────────
scaler = StandardScaler()
scaler.fit(X_train[continuous_cols])

# ── Train all 5 models ───────────────────────────────
models_def = {
    'Linear Regression':  LinearRegression(),
    'Ridge Regression':   Ridge(alpha=1.0, random_state=42),
    'Decision Tree':      DecisionTreeRegressor(max_depth=8, min_samples_split=20, min_samples_leaf=10, random_state=42),
    'Random Forest':      RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=10, min_samples_leaf=5, max_features='sqrt', n_jobs=-1, random_state=42),
    'Gradient Boosting':  GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, min_samples_split=10, subsample=0.8, max_features='sqrt', random_state=42),
}

print("\nTraining models...")
trained = {}
for name, model in models_def.items():
    model.fit(X_train, y_train)
    trained[name] = model
    print(f"  ✅ {name}")

# ── Save bundle — keys match bed_management_app.py ──
bundle = {
    'models':          trained,
    'feature_cols':    list(X_train.columns),
    'continuous_cols': continuous_cols,
    'scaler':          scaler,
    'feature_means':   X_train.mean().to_dict(),
    'feature_stds':    X_train.std().to_dict(),
}

out = base + 'model_bundle.pkl'
with open(out, 'wb') as f:
    pickle.dump(bundle, f)

print(f"\n✅ Saved → model_bundle.pkl")
print(f"   Features  : {len(bundle['feature_cols'])}")
print(f"   Cont. cols: {len(continuous_cols)}")
print(f"   Models    : {list(trained.keys())}")
print("\n✅ Now run:  python -m streamlit run bed_management_app.py")
