import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = joblib.load("model.pkl)
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

y_pred = model.predict(X_test)

print(f"MAE : {mean_absolute_error(y_test, y_pred):,.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
print(f"R²  : {r2_score(y_test, y_pred):.4f}")

# --- Actual vs Predicted Plot ---
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test.iloc[:, 0], y=y_pred, alpha=0.6, color='blue', edgecolor='w')
plt.plot([y_test.min().iloc[0], y_test.max().iloc[0]], [y_test.min().iloc[0], y_test.max().iloc[0]], 'r--', lw=2)
plt.xlabel("Actual Salary (USD)")
plt.ylabel("Predicted Salary (USD)")
plt.title("Actual vs. Predicted Salary")
plt.grid(True)
plt.tight_layout()
plt.show()
