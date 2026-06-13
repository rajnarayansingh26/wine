from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
import joblib

wine = load_wine()

X = wine.data
y = wine.target

model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)

joblib.dump(model, "wine_model.pkl")

print("Model saved successfully")