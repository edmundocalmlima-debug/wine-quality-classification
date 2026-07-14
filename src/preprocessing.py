"""
Funções de pré-processamento para o projeto Wine Quality Classification.
Tech Challenge - Fase 2 - POSTECH
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
QUALITY_THRESHOLD = 7  # nota >= 7 -> alta qualidade


def load_data(path: str) -> pd.DataFrame:
    """Carrega o dataset de vinhos a partir de um CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def create_target(df: pd.DataFrame) -> pd.DataFrame:
    """Cria a variável alvo binária: 1 = alta qualidade (nota >= 7), 0 = baixa/média."""
    df = df.copy()
    df["high_quality"] = (df["quality"] >= QUALITY_THRESHOLD).astype(int)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: cria variáveis derivadas com significado enológico."""
    df = df.copy()
    # Proporção de SO2 livre em relação ao total (quanto do enxofre está "ativo")
    df["free_so2_ratio"] = df["free sulfur dioxide"] / df["total sulfur dioxide"]
    # Interação álcool x sulfatos (dois dos preditores mais fortes)
    df["alcohol_sulphates"] = df["alcohol"] * df["sulphates"]
    # Razão acidez fixa / acidez volátil (equilíbrio de acidez)
    df["acidity_ratio"] = df["fixed acidity"] / (df["volatile acidity"] + 1e-6)
    return df


def split_and_scale(df: pd.DataFrame, use_engineered: bool = True):
    """Separa treino/teste (estratificado) e padroniza as variáveis numéricas."""
    feature_cols = [c for c in df.columns if c not in ("quality", "high_quality")]
    if not use_engineered:
        feature_cols = [c for c in feature_cols
                        if c not in ("free_so2_ratio", "alcohol_sulphates", "acidity_ratio")]

    X = df[feature_cols]
    y = df["high_quality"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_cols, index=X_test.index
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_cols
