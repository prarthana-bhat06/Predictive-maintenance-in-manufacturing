# Predictive-maintenance-in-manufacturing
This project leverages machine learning to classify production efficiency and enable predictive maintenance in smart manufacturing systems using real-time sensor and 6G network data. 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("manufacturing_6G_dataset.csv")

# Step 2: Explore dataset
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())
print(df["Efficiency_Status"].value_counts())

# Step 3: Preprocess
# Drop Timestamp if not useful for prediction
df.drop(columns=["Timestamp"], inplace=True)

# Handle missing values
df.dropna(inplace=True)

# Encode categorical columns
categorical_cols = ["Operation_Mode", "Efficiency_Status"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 4: Feature-target split
X = df.drop(columns=["Efficiency_Status"])
y = df["Efficiency_Status"]

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 7: Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Evaluate
y_pred = model.predict(X_test)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optional: Feature importance
importances = model.feature_importances_
feat_names = X.columns
feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()
![image](https://github.com/user-attachments/assets/f9522668-6717-4710-b6de-3097917222e5)
![image](https://github.com/user-attachments/assets/c1d665e9-452b-4610-ab2f-07f4c96c9523)
![image](https://github.com/user-attachments/assets/c5c6022a-5f93-423a-8d2d-9cb639196996)
![pred-5](https://github.com/user-attachments/assets/a193760f-edfe-483e-82ac-03f489740bd7)


