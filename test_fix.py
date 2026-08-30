import sys
import numpy as np
from src.componenets.model_trainer import ModelTrainer

# Create dummy test data
X_train = np.random.rand(100, 10)
y_train = np.random.rand(100)
X_test = np.random.rand(30, 10)
y_test = np.random.rand(30)

train_array = np.hstack([X_train, y_train.reshape(-1, 1)])
test_array = np.hstack([X_test, y_test.reshape(-1, 1)])

# Test the model trainer
try:
    mt = ModelTrainer()
    r2_score, best_model = mt.initiate_model_trainer(train_array, test_array)
    print(f"✓ SUCCESS! R² Score: {r2_score:.4f}")
    print(f"✓ Best Model Type: {type(best_model).__name__}")
    print("\n✓✓✓ BUG IS FIXED - No KeyError! ✓✓✓")
except KeyError as e:
    print(f"✗ KeyError: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
