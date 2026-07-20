import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load batter features
df = pd.read_csv("batter_features.csv")

# Remove rows with missing values
df = df.dropna()

# Create a fake target for now
# 1 = good hitter, 0 = weaker hitter
df["target"] = (df["avg_exit_velocity"] > 90).astype(int)

# Features
X = df[[
    "avg_exit_velocity",
    "max_exit_velocity",
    "avg_launch_angle",
    "batted_balls"
]]

# Target
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {accuracy:.2f}")

joblib.dump(model, "mlb_model.pkl")

print("Saved model: mlb_model.pkl")