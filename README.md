# 🍷 Wine Quality Classification — Tech Challenge Fase 2 (POSTECH)

Projeto de Machine Learning para **prever a qualidade de vinhos tintos** a partir de suas características físico-químicas, transformando a nota dos especialistas em uma classificação binária:

- **Alta Qualidade**: nota ≥ 7
- **Baixa/Média Qualidade**: nota < 7

## 📊 Resultados principais

| Métrica (teste) | Regressão Logística | Random Forest |
|---|---|---|
| Acurácia | 0,79 | **0,93** |
| Precisão (Alta) | 0,37 | **0,79** |
| Recall (Alta) | **0,80** | 0,63 |
| F1 (Alta) | 0,51 | **0,70** |
| ROC-AUC | 0,86 | **0,92** |

O **Random Forest** foi o modelo vencedor, com melhor equilíbrio entre precisão e recall na classe minoritária e maior poder de discriminação (AUC ≈ 0,92).

**Fatores que mais influenciam a qualidade:** teor alcoólico (+), sulfatos (+) e acidez volátil (−).

## 📁 Estrutura do repositório

```
wine-quality-classification/
│
├── data/                  # Base de dados (winequality-red.csv)
├── notebooks/             # Notebook com a análise completa e outputs
├── src/                   # Scripts auxiliares
│   ├── preprocessing.py   # Carga, alvo binário, feature engineering, split/scale
│   └── run_pipeline.py    # Pipeline completa (EDA -> modelos -> gráficos)
├── results/               # Gráficos e métricas gerados pela pipeline
├── presentation/          # Apresentação executiva (storytelling da EDA)
├── requirements.txt
└── README.md
```

## 🚀 Como executar

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Rodar a pipeline completa (gera gráficos e métricas em results/)
python src/run_pipeline.py

# 3. Ou explorar o notebook
jupyter notebook notebooks/wine_quality_analysis.ipynb
```

## 🔬 Pipeline

1. **Compreensão do problema** — contexto, definição do alvo e binarização da qualidade.
2. **EDA** — distribuições, correlações justificadas, outliers e balanceamento (apenas ~13,6% de vinhos de alta qualidade).
3. **Pré-processamento** — verificação de dados faltantes, padronização (StandardScaler) e feature engineering (`free_so2_ratio`, `alcohol_sulphates`, `acidity_ratio`).
4. **Modelagem** — Regressão Logística (baseline) vs Random Forest, ambos com `class_weight="balanced"` e validação cruzada estratificada de 5 folds.
5. **Avaliação** — matriz de confusão, precisão/recall/F1, curvas ROC.
6. **Interpretação** — importância das variáveis e implicações para o processo produtivo.

## 📦 Dataset

[Wine Quality Dataset](https://www.kaggle.com/datasets/yasserh/wine-quality-dataset) — 1.599 amostras de vinho tinto, 11 variáveis físico-químicas + nota de qualidade (especialistas).
