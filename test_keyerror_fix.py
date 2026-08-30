import sys
import numpy as np
from src.utils import evaluate_models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Create test data with better scores
np.random.seed(42)
X_train = np.random.rand(200, 10)
y_train = 5 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(200) * 0.1
X_test = np.random.rand(60, 10)
y_test = 5 * X_test[:, 0] + 3 * X_test[:, 1] + np.random.randn(60) * 0.1

# Define models - MATCHING THE KEYS IN MODEL_TRAINER.PY
models = {
    "Random Forest": RandomForestRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Linear Regression": LinearRegression(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "XGBRegressor": XGBRegressor(verbosity=0),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
}

# Define params - MATCHING THE KEYS IN MODEL_TRAINER.PY
params = {
    "Decision Tree": {
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    },
    "Random Forest": {
        'n_estimators': [8, 16, 32]
    },
    "Gradient Boosting": {
        'learning_rate': [.1, .01],
        'n_estimators': [8, 16]
    },
    "Linear Regression": {},
    "KNeighborsRegressor": {},
    "XGBRegressor": {
        'learning_rate': [.1, .01],
        'n_estimators': [8, 16]
    },
    "CatBoosting Regressor": {
        'depth': [6, 8],
        'learning_rate': [0.01, 0.05],
        'iterations': [30, 50]
    },
    "AdaBoost Regressor": {
        'learning_rate': [.1, .01],
        'n_estimators': [8, 16]
    }
}

print("Testing evaluate_models function...")
print(f"Models keys: {list(models.keys())}")
print(f"Params keys:  {list(params.keys())}")
print("\nChecking key matching...")

# Verify all model keys exist in params
try:
    for model_key in models.keys():
        if model_key not in params:
            print(f"✗ MISMATCH: '{model_key}' in models but NOT in params")
            sys.exit(1)
        else:
            print(f"✓ '{model_key}' found in both dicts")
    
    print("\n✓ All keys match! Now testing evaluate_models()...")
    
    # Run evaluate_models
    report = evaluate_models(X_train, y_train, X_test, y_test, models, params)
    
    print("\n✓✓✓ SUCCESS! No KeyError! ✓✓✓")
    print(f"\nModel Performance Report:")
    for model_name, score in report.items():
        print(f"  {model_name}: R² = {score:.4f}")
        
except KeyError as e:
    print(f"\n✗✗✗ KeyError STILL EXISTS: {e} ✗✗✗")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
