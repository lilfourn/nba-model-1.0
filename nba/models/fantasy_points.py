import pandas as pd
import sqlite3
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import math
import os

# Get absolute path to database
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_path = os.path.join(base_dir, "data", "cleaned_playerGameStats.db")

# Connect to the database
conn = sqlite3.connect(db_path)
df = pd.read_sql_query("SELECT * FROM cleaned_game_stats", conn)
conn.close()

# Extract date from game_id and convert to datetime
df['game_date'] = pd.to_datetime(df['game_id'].str[:8], format='%Y%m%d')
df = df.sort_values('game_date')

# Define forecast column and parameters
forecast_col = 'fantasy_points'
forecast_out = int(math.ceil(0.00001 * len(df)))  # Predicting 10% of dataset length into future
print(forecast_out)

# Select features for prediction (excluding non-numeric and target columns)
features = ['blocks', 'offensive_rebounds', 'defensive_rebounds', 'total_rebounds',
           'assists', 'steals', 'points', 'turnovers', 'field_goals_attempted',
           'field_goals_made', 'field_goal_percentage', 'three_point_made',
           'three_point_attempted', 'three_point_percentage', 'free_throws_attempted',
           'free_throws_made', 'free_throw_percentage', 'personal_fouls',
           'plus_minus', 'minutes_played']

# Prepare features (X) and target (y)
X = df[features].fillna(-99999)
y = df[forecast_col]

# Create future target by shifting
df['label'] = df[forecast_col].shift(-forecast_out)

# Remove rows with NaN labels
df.dropna(inplace=True)

# Scale features
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data chronologically - use the last 20% of the data (by date) as test set
split_idx = int(len(df) * 0.8)
X_train = X_scaled[:split_idx]
X_test = X_scaled[split_idx:]
y_train = y[:split_idx]
y_test = y[split_idx:]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Print model accuracy
accuracy = model.score(X_test, y_test)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print feature importance
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': abs(model.coef_)
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending = False))
