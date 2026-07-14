"""
Pipeline completa: EDA -> pré-processamento -> modelagem -> avaliação -> interpretação.
Gera todos os gráficos e métricas na pasta results/.

Tech Challenge - Fase 2 - POSTECH
Uso: python src/run_pipeline.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_score

sys.path.append(os.path.dirname(__file__))
from preprocessing import (RANDOM_STATE, add_features, create_target,
                           load_data, split_and_scale)

sns.set_theme(style="whitegrid", palette="deep")

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(BASE_DIR, "data", "winequality-red.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def save(fig, name):
    fig.savefig(os.path.join(RESULTS_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------
# 1. Carga e compreensão do problema
# ---------------------------------------------------------------
df = load_data(DATA_PATH)
print(f"Dataset: {df.shape[0]} linhas x {df.shape[1]} colunas")
print(f"Valores faltantes: {int(df.isna().sum().sum())}")
print(f"Linhas duplicadas: {int(df.duplicated().sum())}")

df = create_target(df)

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
# 2.1 Distribuição da nota de qualidade e balanceamento das classes
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.countplot(x="quality", data=df, ax=axes[0], color="#4c72b0")
axes[0].set_title("Distribuição da nota de qualidade")
axes[0].set_xlabel("Nota (especialistas)")
axes[0].set_ylabel("Nº de amostras")

class_counts = df["high_quality"].value_counts().sort_index()
axes[1].bar(["Baixa/Média (<7)", "Alta (>=7)"], class_counts.values,
            color=["#4c72b0", "#c44e52"])
for i, v in enumerate(class_counts.values):
    axes[1].text(i, v + 15, f"{v}\n({v / len(df):.1%})", ha="center")
axes[1].set_title("Balanceamento das classes (alvo binário)")
axes[1].set_ylabel("Nº de amostras")
axes[1].set_ylim(0, class_counts.max() * 1.2)
save(fig, "01_distribuicao_classes.png")

# 2.2 Distribuição das variáveis físico-químicas
num_cols = [c for c in df.columns if c not in ("quality", "high_quality")]
fig, axes = plt.subplots(3, 4, figsize=(16, 10))
for ax, col in zip(axes.flat, num_cols):
    sns.histplot(df[col], kde=True, ax=ax, color="#4c72b0")
    ax.set_title(col)
    ax.set_xlabel("")
for ax in axes.flat[len(num_cols):]:
    ax.axis("off")
fig.suptitle("Distribuição das variáveis físico-químicas", y=1.01, fontsize=14)
fig.tight_layout()
save(fig, "02_distribuicoes_variaveis.png")

# 2.3 Matriz de correlação
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[num_cols + ["quality"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            ax=ax, annot_kws={"size": 8})
ax.set_title("Matriz de correlação (Pearson)")
save(fig, "03_matriz_correlacao.png")

corr_target = corr["quality"].drop("quality").sort_values()
print("\nCorrelação com a qualidade:")
print(corr_target.round(3).to_string())

# 2.4 Boxplots das variáveis mais correlacionadas vs classe
top_vars = corr_target.abs().sort_values(ascending=False).head(4).index.tolist()
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
for ax, col in zip(axes, top_vars):
    sns.boxplot(x="high_quality", y=col, data=df, ax=ax,
                hue="high_quality", palette=["#4c72b0", "#c44e52"], legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baixa/Média", "Alta"])
    ax.set_xlabel("")
    ax.set_title(col)
fig.suptitle("Variáveis mais correlacionadas vs. classe de qualidade", y=1.03)
fig.tight_layout()
save(fig, "04_boxplots_top_variaveis.png")

# 2.5 Outliers (regra IQR)
outlier_summary = {}
for col in num_cols:
    q1, q3 = df[col].quantile([0.25, 0.75])
    iqr = q3 - q1
    mask = (df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)
    outlier_summary[col] = int(mask.sum())
print("\nOutliers por variável (regra 1,5x IQR):")
print(pd.Series(outlier_summary).sort_values(ascending=False).to_string())

# ---------------------------------------------------------------
# 3. Pré-processamento (feature engineering + split + padronização)
# ---------------------------------------------------------------
df_feat = add_features(df)
X_train, X_test, y_train, y_test, scaler, feature_cols = split_and_scale(df_feat)
print(f"\nTreino: {X_train.shape} | Teste: {X_test.shape}")
print(f"Features utilizadas ({len(feature_cols)}): {feature_cols}")

# ---------------------------------------------------------------
# 4. Modelagem: Regressão Logística vs Random Forest
# ---------------------------------------------------------------
models = {
    "Regressão Logística": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=400, class_weight="balanced",
        min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = {}
probas = {}

for name, model in models.items():
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    probas[name] = y_proba

    results[name] = {
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    print(f"\n===== {name} =====")
    print(f"F1 (validação cruzada 5-fold): {cv_f1.mean():.3f} +/- {cv_f1.std():.3f}")
    print(classification_report(y_test, y_pred,
                                target_names=["Baixa/Média", "Alta"]))

# ---------------------------------------------------------------
# 5. Avaliação: matrizes de confusão, ROC e comparação
# ---------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, res) in zip(axes, results.items()):
    cm = np.array(res["confusion_matrix"])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Baixa/Média", "Alta"],
                yticklabels=["Baixa/Média", "Alta"])
    ax.set_title(f"{name}\nMatriz de confusão (teste)")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
fig.tight_layout()
save(fig, "05_matrizes_confusao.png")

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for name, y_proba in probas.items():
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    ax.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['roc_auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
ax.set_xlabel("Taxa de falsos positivos")
ax.set_ylabel("Taxa de verdadeiros positivos")
ax.set_title("Curvas ROC - conjunto de teste")
ax.legend()
save(fig, "06_curvas_roc.png")

metrics_df = pd.DataFrame(results).T[
    ["accuracy", "precision", "recall", "f1", "roc_auc"]
]
fig, ax = plt.subplots(figsize=(9, 4.5))
metrics_df.plot(kind="bar", ax=ax, rot=0,
                color=["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3"])
ax.set_title("Comparação de métricas entre os modelos (teste)")
ax.set_ylim(0, 1)
ax.legend(loc="lower right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8)
save(fig, "07_comparacao_metricas.png")

# ---------------------------------------------------------------
# 6. Interpretação: importância das variáveis
# ---------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
importances.plot(kind="barh", ax=ax, color="#4c72b0")
ax.set_title("Importância das variáveis - Random Forest")
ax.set_xlabel("Importância (redução média de impureza)")
save(fig, "08_importancia_variaveis.png")

# Coeficientes da regressão logística (dados padronizados -> comparáveis)
lr = models["Regressão Logística"]
coefs = pd.Series(lr.coef_[0], index=feature_cols).sort_values()
fig, ax = plt.subplots(figsize=(8, 6))
colors = ["#c44e52" if v < 0 else "#55a868" for v in coefs]
coefs.plot(kind="barh", ax=ax, color=colors)
ax.set_title("Coeficientes padronizados - Regressão Logística")
ax.set_xlabel("Efeito na chance de 'Alta Qualidade'")
ax.axvline(0, color="k", lw=0.8)
save(fig, "09_coeficientes_logistica.png")

# ---------------------------------------------------------------
# Exportar métricas
# ---------------------------------------------------------------
with open(os.path.join(RESULTS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
metrics_df.round(4).to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"))

print("\nPipeline concluída. Gráficos e métricas salvos em results/.")
