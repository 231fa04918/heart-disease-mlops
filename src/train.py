import yaml, json, pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

params = yaml.safe_load(open("params.yaml"))

df = pd.read_csv("data/heart.csv").dropna()
X = df.drop(columns=["num"])
X = pd.get_dummies(X, drop_first=True)
y = (df["num"] > 0).astype(int)   # binary: disease present or not

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=params["train"]["test_size"],
    random_state=params["train"]["random_state"]
)

model = RandomForestClassifier(
    n_estimators=params["model"]["n_estimators"],
    max_depth=params["model"]["max_depth"],
    min_samples_split=params["model"]["min_samples_split"],
    random_state=params["model"]["random_state"]
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

metrics = {
    "accuracy": accuracy_score(y_test, preds),
    "precision": precision_score(y_test, preds),
    "recall": recall_score(y_test, preds),
    "f1_score": f1_score(y_test, preds)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print(metrics)