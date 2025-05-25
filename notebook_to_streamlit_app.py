import streamlit as st

st.title("Converted Jupyter Notebook to Streamlit")

# --- Code from Notebook ---
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 

df = pd.read_csv("TrKMgk.csv")

df

df.head()

df.describe()

df.info()

df.isnull().sum()

df.dropna()

df['success']

df = pd.get_dummies(df, columns=["rocket_name", "orbit", "site_name", "location"], drop_first=True)

df

X = df.drop(["mission_name", "launch_date", "success"], axis=1)
y = df["success"].map({"True": 1, "False": 0})

X

y

y.fillna(0, inplace=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
import seaborn as sns

importances = model.feature_importances_
features = X.columns

sns.barplot(x=importances, y=features)
plt.title("Feature Importance")
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
df = pd.read_csv("TrKMgk.csv")
df = pd.get_dummies(df, columns=["rocket_name", "orbit", "site_name", "location"], drop_first=True)
df["success"] = df["success"].astype(str).map({"True": 1, "False": 0})
df = df.dropna(subset=["success"])
X = df.drop(["mission_name", "launch_date", "success"], axis=1)
y = df["success"]
X = X.fillna(X.mean(numeric_only=True))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
importances = model.feature_importances_
features = X.columns
top_features = pd.Series(importances, index=features).sort_values(ascending=False).head(10)


plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()


