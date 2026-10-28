import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataset_config import DATASETS


def train_dataset(dataset):
    df = pd.read_csv(dataset["path"])
    if "heart.csv" in dataset["path"]:
        df.loc[df["Cholesterol"] == 0, "Cholesterol"] = pd.NA
        df.loc[df["RestingBP"] == 0, "RestingBP"] = pd.NA
    features = dataset["numeric"] + dataset["categorical"]
    X, y = df[features], df[dataset["target"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocess = ColumnTransformer([
        (
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            dataset["numeric"],
        ),
        (
            "cat",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]),
            dataset["categorical"],
        ),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42)),
    ])
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    print(f"\n{dataset['label']}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions, zero_division=0))
    return model


if __name__ == "__main__":
    models = {
        key: train_dataset(dataset)
        for key, dataset in DATASETS.items()
    }
    joblib.dump(models, "models.joblib", compress=3)
