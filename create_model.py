import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

df = pd.read_csv("mental_health_data.csv")

y_dep = df['depression']
y_anx = df['anxiety']
X = df.drop(['depression', 'anxiety'], axis=1)

X_encoded = pd.get_dummies(X)

feature_cols = X_encoded.columns.tolist()

# Train models
model_dep = RandomForestClassifier()
model_dep.fit(X_encoded, y_dep)

model_anx = RandomForestClassifier()
model_anx.fit(X_encoded, y_anx)

# Save models and features
with open('mental_health_model.pkl', 'wb') as f:
    pickle.dump((model_dep, model_anx, feature_cols), f)

print("Model trained & saved successfully with one-hot encoding.")
