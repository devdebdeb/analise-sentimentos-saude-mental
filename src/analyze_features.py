import joblib
import pandas as pd
import os

MODEL_PATH = 'modelo_final_evasao.joblib'
BASE_DATA_PATH = 'dataset_brasil_base.csv'
SCORES_DATA_PATH = 'resultados_saude_mental.csv'

# Código de Preparação (Repetido do classificador)
df_base = pd.read_csv(BASE_DATA_PATH)
df_scores = pd.read_csv(SCORES_DATA_PATH, sep=';', decimal=',') 
df_scores = df_scores.rename(columns={'id_texto': 'student_id'})
df_final = pd.merge(df_base, df_scores[['student_id', 'score_ansiedade', 'score_depressao', 'score_burnout']], on='student_id', how='left')
df_final.dropna(subset=['score_ansiedade'], inplace=True)

# Lista das features usadas no treinamento
features = [
    'idade', 'horas_sono', 'exercicios_semana', 'horas_tela_dia', 'cafeina_mg_dia', 
    'faltas_mensais', 'notas_periodo', 
    'autoavaliação_stress', 'autoavaliação_ansiedade', 'autoavaliação_felicidade', 
    'faz_terapia', 'doença_crônica', 'usa_medicação', 'gênero', 'curso', 'vive_com', 'renda_familiar',
    'score_ansiedade', 'score_depressao', 'score_burnout'
]
X = df_final[features]
categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

try:
    # 1. Carregar o modelo
    model = joblib.load(MODEL_PATH)
    
    # 2. Extrair a importância
    importances = model.feature_importances_
    
    # 3. Criar DataFrame de resultados
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importances.head(10).to_markdown(index=False))
    
except FileNotFoundError:
    print(f"Erro: Modelo '{MODEL_PATH}' não encontrado. Execute o treinamento primeiro.")