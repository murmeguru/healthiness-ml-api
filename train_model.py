import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Sample data
data = {
    'sugar': [10, 5, 20, 2, 1],
    'fat': [5, 2, 10, 1, 0],
    'salt': [1, 0.5, 2, 0.2, 0.1],
    'energy': [200, 100, 300, 80, 60],
    'health_score': [30, 70, 20, 90, 95]
}
df = pd.DataFrame(data)

X = df[['sugar', 'fat', 'salt', 'energy']]
y = df['health_score']

model = RandomForestRegressor()
model.fit(X, y)

joblib.dump(model, 'health_model.pkl')

print("✅ Model trained and saved!")
