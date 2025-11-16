import pandas as pd
import joblib
import os
import numpy as np

# --- 🚨 PARÂMETROS CHAVE DE INTERVENÇÃO 🚨 ---
LIMIAR_ALERTA = 0.19 # Limiar de Alto Recall (~72%)
LIMIAR_CRITICO = 0.50 # Limiar de Alta Precisão (Risco Crítico > 50% chance)
# ----------------------------------------------

def categorizar_risco(prob: float) -> str:
    """Classifica a probabilidade de evasão em 3 níveis de prioridade."""
    if prob >= LIMIAR_CRITICO:
        return 'PRIORIDADE 1: CRÍTICO (Intervenção Imediata)'
    elif prob >= LIMIAR_ALERTA:
        return 'PRIORIDADE 2: ALTO (Monitoramento Ativo)'
    else:
        return 'PRIORIDADE 3: PADRÃO (Monitoramento de Rotina)'


def predict_final_dropout_risk(input_base_path: str, input_scores_path: str, model_path: str, scaler_path: str):
    """
    Carrega o modelo XGBoost final e o scaler, junta os dados,
    aplica pré-processamento e gera a previsão BINÁRIA e PRIORIZADA de evasão.
    """
    print("--- INICIANDO PREDIÇÃO FINAL DE EVASÃO ---")

    # --- 1. Carregar Modelo e Scaler ---
    print("\n[ETAPA 1/5] Carregando modelo final e scaler...")
    try:
        model_evasion = joblib.load(model_path)
        # Note: O scaler não é usado pelo XGBoost, mas é carregado por segurança
        scaler = joblib.load(scaler_path) 
        print("Modelo e Scaler carregados com sucesso!")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}. Certifique-se de ter rodado 'train_final_classifier.py' primeiro.")
        return

    # --- 2. Carregar e Combinar os Datasets ---
    print("\n[ETAPA 2/5] Combinando dados base e scores NLP...")
    try:
        df_base = pd.read_csv(input_base_path)
        df_scores = pd.read_csv(input_scores_path, sep=';', decimal=',')
    except FileNotFoundError as e:
        print(f"Erro: Dados de entrada não encontrados - {e}.")
        return

    df_scores = df_scores.rename(columns={'id_texto': 'student_id'})
    df_final = pd.merge(
        df_base, 
        df_scores[['student_id', 'score_ansiedade', 'score_depressao', 'score_burnout']], 
        on='student_id', 
        how='left'
    )
    df_final.dropna(subset=['score_ansiedade'], inplace=True)
    
    # --- 3. Feature Engineering & Pré-processamento (IDÊNTICO AO TREINAMENTO) ---
    print("\n[ETAPA 3/5] Aplicando Feature Engineering e One-Hot Encoding...")
    
    # Recria as features de interação
    df_final['risco_burnout_comportamental'] = (
        df_final['score_burnout'] * df_final['horas_tela_dia'] / (df_final['horas_sono'] + 0.1)
    )
    df_final['impacto_depressao_notas'] = df_final['score_depressao'] / (df_final['notas_periodo'] + 0.1)
    df_final['vulnerabilidade_academica'] = df_final['score_ansiedade'] * df_final['faltas_mensais']
    
    # Define as features originais (deve ser a mesma lista do treinamento)
    features = [
        'idade', 'horas_sono', 'exercicios_semana', 'horas_tela_dia', 'cafeina_mg_dia', 
        'faltas_mensais', 'notas_periodo', 
        'risco_burnout_comportamental', 'impacto_depressao_notas', 'vulnerabilidade_academica',
        'autoavaliação_stress', 'autoavaliação_ansiedade', 'autoavaliação_felicidade', 
        'score_ansiedade', 'score_depressao', 'score_burnout',
        'faz_terapia', 'doença_crônica', 'usa_medicação', 'gênero', 'curso', 'vive_com', 'renda_familiar',
    ]

    X = df_final[features]
    categorical_cols = X.select_dtypes(include=['object']).columns
    X_processed = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    # --- 4. Predição com o modelo (Obtendo a Probabilidade) ---
    print("\n[ETAPA 4/5] Calculando probabilidades de evasão...")
    
    # Obtém a probabilidade para a classe Dropout (índice 1)
    y_proba = model_evasion.predict_proba(X_processed.values)[:, 1] 
    
    # --- 5. Aplicação da Priorização Otimizada e Resultados ---
    print(f"\n[ETAPA 5/5] Aplicando Limiares de Prioridade (Crítico: {LIMIAR_CRITICO}, Alerta: {LIMIAR_ALERTA})...")

    df_final['probabilidade_dropout'] = y_proba
    # Aplica a função de categorização a cada probabilidade
    df_final['risco_intervencao'] = df_final['probabilidade_dropout'].apply(categorizar_risco)
    
    colunas_resultado = ['student_id', 'probabilidade_dropout', 'risco_intervencao']
    df_resultado = df_final[colunas_resultado].round(4)
    
    print("\n--- RELATÓRIO DE INTERVENÇÃO (PRIORIZADO) ---")
    print(df_resultado.head(10).to_markdown(index=False))
    
    # Resumo da Contagem por Prioridade
    contagem = df_resultado['risco_intervencao'].value_counts()
    
    print("\n--- RESUMO DE ALERTAS POR PRIORIDADE ---")
    print(contagem.to_markdown())


if __name__ == "__main__":
    # Define os nomes dos arquivos de entrada e saída
    INPUT_BASE = 'data/dataset_brasil_base.csv'
    INPUT_SCORES = 'data/resultados_saude_mental.csv'
    MODEL_PATH = 'modelos_finais/modelo_final_MAX_PERFORMANCE.joblib'
    SCALER_PATH = 'modelos_finais/scaler_final.joblib'
    
    # Adiciona a nova função de categorização antes de rodar o script
    predict_final_dropout_risk(
        input_base_path=INPUT_BASE, 
        input_scores_path=INPUT_SCORES, 
        model_path=MODEL_PATH, 
        scaler_path=SCALER_PATH
    )