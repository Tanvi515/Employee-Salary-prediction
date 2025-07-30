import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv("/content/salaries.csv")
print(df.info())
print(df.job_title.value_counts())
print(df.employee_residence.value_counts())
print(df.company_location.value_counts())

import matplotlib
import matplotlib.pyplot as plt
plt.boxplot(df.salary_in_usd)
plt.show()

df = preprocess_data(df)
X = df.drop(columns=["salary_in_usd"])
y = df["salary_in_usd"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor()
model.fit(X_train, y_train)

joblib.dump(model, "model.pkl")
X_test.to_csv("X_test.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
