import sys
import numpy as np
from src.componenets.model_trainer import ModelTrainer

print("Testing actual ModelTrainer with fixed parameters...")

# Create test data with better scores for real training
np.random.seed(42)
X_train = np.random.rand(200, 10)
y_train = 5 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(200) * 0.1
X_test = np.random.rand(60, 10)
y_test = 5 * X_test[:, 0] + 3 * X_test[:, 1] + np.random.randn(60) * 0.1

train_array = np.hstack([X_train, y_train.reshape(-1, 1)])
test_array = np.hstack([X_test, y_test.reshape(-1, 1)])

try:
    mt = ModelTrainer()
    print("Calling initiate_model_trainer()...")
    r2_score, best_model = mt.initiate_model_trainer(train_array, test_array)
    print(f"\n✓✓✓ SUCCESS! ✓✓✓")
    print(f"Best Model: {type(best_model).__name__}")
    print(f"R² Score: {r2_score:.4f}")
except ValueError as e:
    if "friedman_mse" in str(e):
        print(f"✗ DEPRECATED PARAMETER ERROR: {e}")
        sys.exit(1)
    else:
        raise
except KeyError as e:
    print(f"✗ KeyError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
