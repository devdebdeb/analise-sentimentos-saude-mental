import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Importa nossa nova função para gerar embeddings
from feature_extractor_embeddings import gerar_embedding

# --- 1. Carregamento dos Dados ---
print("Carregando o dataset simulado...")
df = pd.read_csv('diarios_universitarios.csv')

# --- 2. Extração de Features (AGORA COM EMBEDDINGS) ---
print("\nIniciando geração de embeddings para cada texto... (Isso vai levar um tempo)")

# A função .apply() vai chamar 'gerar_embedding' para cada linha da coluna 'diary_text'
# O resultado será uma Series do Pandas onde cada item é um vetor (array NumPy)
embeddings = df['diary_text'].apply(gerar_embedding)

# Removemos qualquer entrada que possa ter falhado
embeddings = embeddings.dropna()
df_filtrado = df.loc[embeddings.index]

# Convertendo a lista de vetores em um formato que o Scikit-learn entende (uma matriz 2D)
X = np.vstack(embeddings.values)

print("Embeddings gerados com sucesso!")
print(f"Nossa matriz de features (X) tem o formato: {X.shape}") # Ex: (164, 768)

# --- 3. Preparação para o Treinamento ---
print("\nPreparando dados para o modelo...")

# Nossa variável alvo (y) continua a mesma
y = df_filtrado['ground_truth_label']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# --- 4. Treinamento do Modelo ---
print("\nTreinando o modelo RandomForestClassifier com features de embedding...")

# Vamos usar nosso melhor modelo anterior (RandomForest) com os parâmetros otimizados que encontramos
# para ver como ele se comporta com as novas features.
modelo = RandomForestClassifier(
    n_estimators=398,
    min_samples_split=17,
    min_samples_leaf=6,
    max_features='sqrt',
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
modelo.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

# --- 5. Avaliação do Modelo ---
print("\n--- AVALIAÇÃO DO MODELO COM EMBEDDINGS ---")

y_pred = modelo.predict(X_test)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

print("\nRelatório de Classificação:")
print(classification_report(y_test_labels, y_pred_labels, zero_division=0))
print(f"Acurácia Geral: {accuracy_score(y_test, y_pred):.2f}")