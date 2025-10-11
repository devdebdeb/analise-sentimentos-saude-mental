import pandas as pd
import joblib
import os
import numpy as np
from feature_extractor_embeddings import gerar_embedding
from sklearn.ensemble import RandomForestClassifier

def train_and_save_models():
    """
    Orquestra o pipeline de treinamento: carrega dados, treina os 3 modelos
    especializados e os salva em disco para uso futuro.
    """
    print("--- INICIANDO SESSÃO DE TREINAMENTO ---")

    # --- 1. Carregamento e Geração de Features (Embeddings) ---
    print("\n[ETAPA 1/3] Carregando dataset e gerando embeddings...")
    try:
        df = pd.read_csv('diarios_universitarios.csv')
    except FileNotFoundError:
        print("Erro: Arquivo 'diarios_universitarios.csv' não encontrado. O treinamento não pode continuar.")
        return
    
    embeddings = df['diary_text'].apply(gerar_embedding).dropna()
    df_processado = df.loc[embeddings.index].copy()
    X = np.vstack(embeddings.values)
    y_original = df_processado['ground_truth_label']
    print(f"Processamento de {len(df_processado)} textos concluído.")

    # --- 2. Preparação dos Alvos e Treinamento dos Modelos ---
    print("\n[ETAPA 2/3] Treinando 3 modelos especializados...")
    
    df_processado['is_anxiety'] = y_original.isin(['risco_ansiedade', 'risco_misto']).astype(int)
    df_processado['is_depression'] = y_original.isin(['risco_depressao', 'risco_misto']).astype(int)
    df_processado['is_burnout'] = y_original.isin(['risco_burnout']).astype(int)
    
    modelos = {}
    condicoes = ['anxiety', 'depression', 'burnout']
    
    for condicao in condicoes:
        print(f"  - Treinando modelo para: {condicao.upper()}...")
        y = df_processado[f'is_{condicao}']
        
        # Usamos nosso melhor modelo (RandomForest) com os parâmetros otimizados.
        modelo = RandomForestClassifier(
            n_estimators=398, min_samples_split=17, min_samples_leaf=6,
            max_features='sqrt', max_depth=None, random_state=42,
            class_weight='balanced', n_jobs=-1
        )
        
        # Treinamos o modelo com 100% dos dados para obter a melhor performance.
        modelo.fit(X, y)
        modelos[condicao] = modelo

    print("Modelos treinados com sucesso!")

    # --- 3. Salvando os Modelos em Disco ---
    print("\n[ETAPA 3/3] Salvando modelos na pasta 'modelos_finais'...")
    os.makedirs('modelos_finais', exist_ok=True)
    for condicao, modelo in modelos.items():
        caminho_arquivo = os.path.join('modelos_finais', f'modelo_{condicao}.joblib')
        joblib.dump(modelo, caminho_arquivo)
        print(f"  - Modelo salvo em: {caminho_arquivo}")

    print("\n--- SESSÃO DE TREINAMENTO CONCLUÍDA COM SUCESSO ---")

if __name__ == "__main__":
    train_and_save_models()