"""Data loading and preprocessing for the IBM Telco Customer Churn dataset."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
TARGET = "Churn"


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges is blank for 11 brand-new customers (tenure == 0) -> 0.0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    df["ChargeDelta"] = df["MonthlyCharges"] - df["AvgMonthlySpend"]
    services = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    df["NumServices"] = (df[services] == "Yes").sum(axis=1)
    df["LogTenure"] = np.log1p(df["tenure"])
    return df


ENGINEERED = ["AvgMonthlySpend", "ChargeDelta", "NumServices", "LogTenure"]
CONT_COLS = NUMERIC + ENGINEERED + BINARY


class Preprocessor:
    """Scales continuous features and integer-encodes categoricals for embeddings."""

    def fit(self, df: pd.DataFrame):
        self.scaler = StandardScaler().fit(df[CONT_COLS].values)
        self.vocab = {c: {v: i + 1 for i, v in enumerate(sorted(df[c].unique()))}
                      for c in CATEGORICAL}  # 0 reserved for unseen
        return self

    def transform(self, df: pd.DataFrame):
        x_cont = self.scaler.transform(df[CONT_COLS].values).astype(np.float32)
        x_cat = np.stack([df[c].map(self.vocab[c]).fillna(0).astype(int).values
                          for c in CATEGORICAL], axis=1).astype(np.int64)
        return x_cont, x_cat

    @property
    def cardinalities(self):
        return [len(self.vocab[c]) + 1 for c in CATEGORICAL]


def split(df: pd.DataFrame, seed: int = 42):
    """Stratified 70 / 15 / 15 train / validation / test split."""
    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=seed)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
