import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score
import os
import sys

# --- 1. DEFINIÇÕES DE CAMINHO E MODELO ---
MODEL_OUTPUT_PATH = 'modelo_final_MAX_PERFORMANCE.joblib'
BASE_DATA_PATH = 'dataset_brasil_base.csv'
SCORES_DATA_PATH = 'resultados_saude_mental.csv'

try:
    # Carregar o modelo otimizado (XGBoost)
    model = joblib.load(MODEL_OUTPUT_PATH)
    print(f"Modelo '{MODEL_OUTPUT_PATH}' carregado com sucesso.")
except FileNotFoundError:
    print(f"Erro: Modelo '{MODEL_OUTPUT_PATH}' não encontrado. Execute o treinamento antes.")
    sys.exit()

# --- 2. RECRIAÇÃO DO CONJUNTO DE TESTE (X_test e y_test) ---
print("Recarregando dados de teste...")

# Carregar e Combinar os Datasets
df_base = pd.read_csv(BASE_DATA_PATH)
df_scores = pd.read_csv(SCORES_DATA_PATH, sep=';', decimal=',') 
df_scores = df_scores.rename(columns={'id_texto': 'student_id'})

df_final = pd.merge(
    df_base, 
    df_scores[['student_id', 'score_ansiedade', 'score_depressao', 'score_burnout']], 
    on='student_id', 
    how='left'
)
df_final.dropna(subset=['score_ansiedade'], inplace=True)

# ----------------------------------------------------------------
# ETAPA DE FEATURE ENGINEERING AVANÇADO (DEVE SER IDÊNTICA AO TREINAMENTO)
# ----------------------------------------------------------------
df_final['risco_burnout_comportamental'] = (
    df_final['score_burnout'] * df_final['horas_tela_dia'] / (df_final['horas_sono'] + 0.1)
)
df_final['impacto_depressao_notas'] = df_final['score_depressao'] / (df_final['notas_periodo'] + 0.1)
df_final['vulnerabilidade_academica'] = df_final['score_ansiedade'] * df_final['faltas_mensais']

# ----------------------------------------------------------------
# PREPARAÇÃO FINAL DE FEATURES (X) E ALVO (Y)
# ----------------------------------------------------------------
features = [
    'idade', 'horas_sono', 'exercicios_semana', 'horas_tela_dia', 'cafeina_mg_dia', 
    'faltas_mensais', 'notas_periodo', 
    'risco_burnout_comportamental', 'impacto_depressao_notas', 'vulnerabilidade_academica',
    'autoavaliação_stress', 'autoavaliação_ansiedade', 'autoavaliação_felicidade', 
    'score_ansiedade', 'score_depressao', 'score_burnout',
    'faz_terapia', 'doença_crônica', 'usa_medicação', 'gênero', 'curso', 'vive_com', 'renda_familiar',
]

X = df_final[features]
y = df_final['final_target'] 
y_encoded = y.replace({'Graduate': 0, 'Dropout': 1}) # 1 = Dropout

# One-Hot Encoding
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# ----------------------------------------------------------------
# RECRIAÇÃO DO SPLIT (random_state=42 é a chave para a identidade do X_test)
# ----------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)
print("Variáveis X_test e y_test recarregadas com sucesso.")

# --- 3. FUNÇÃO DE OTIMIZAÇÃO DE LIMIAR ---
def find_best_threshold(model, X_test, y_test, target_class=1):
    """Calcula as métricas para diferentes limiares de probabilidade."""
    
    # Obtém as probabilidades para a classe positiva (Dropout = 1)
    y_proba = model.predict_proba(X_test)[:, target_class]
    
    # Testa limiares de 10% a 50%
    limiares = np.arange(0.10, 0.50, 0.01) 
    
    resultados = []
    melhor_recall = 0
    melhor_limiar = 0.5
    
    for limiar in limiares:
        y_pred = (y_proba >= limiar).astype(int)
        
        recall = recall_score(y_test, y_pred, pos_label=target_class, zero_division=0)
        precision = precision_score(y_test, y_pred, pos_label=target_class, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=target_class, zero_division=0)
        
        # Otimiza o melhor limiar com base no Recall (prioridade de alerta)
        if recall > melhor_recall:
            melhor_recall = recall
            melhor_limiar = limiar

        resultados.append({
            'Limiar': limiar,
            'Recall': recall,
            'Precision': precision,
            'F1-Score': f1
        })

    df_resultados = pd.DataFrame(resultados).round(4)
    return df_resultados, melhor_limiar

# --- 4. EXECUÇÃO DA OTIMIZAÇÃO ---
df_otimizacao, limiar_otimo_recall = find_best_threshold(model, X_test, y_test, target_class=1)

print(f"\n--- Otimização de Limiar para MÁXIMO RECALL de Dropout ---")
print(f"O limiar que gera o MAIOR RECALL é: {limiar_otimo_recall:.4f}")

# Exibir os resultados para análise de trade-off
# Filtramos a tabela para mostrar Recall >= 0.60 para visualizarmos a intervenção
df_intervencao = df_otimizacao[df_otimizacao['Recall'] >= 0.60].sort_values(by='Recall', ascending=False)


print("\n--- ANÁLISE DE TRADE-OFF: MÁXIMO RECALL (INTERVENÇÃO) ---")

if df_intervencao.empty:
    # Se não atingimos 60% de Recall, mostramos os 5 melhores resultados
    df_intervencao = df_otimizacao.sort_values(by='Recall', ascending=False).head(10)
    print("Não foi possível atingir Recall de 0.60. Melhores resultados:")
else:
    print("Melhores Limiares para intervenção (Recall >= 0.60):")

print(df_intervencao.to_markdown(index=False))

print("\n--- CONCLUSÃO E RECOMENDAÇÃO ---")
print("Para um SISTEMA DE ALERTA PRECOCE, o foco deve ser o Recall.")
print("Um limiar mais baixo (ex: 0.20 ou 0.25) aumenta o Recall (captura mais alunos em risco),")
print("mas diminui a Precision (mais falsos alertas).")