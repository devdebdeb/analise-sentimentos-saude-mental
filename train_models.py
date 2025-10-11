import pandas as pd
import joblib
import os
import numpy as np
from feature_extractor_embeddings import gerar_embedding
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV # <-- NOVO IMPORT

def train_and_save_models():
    """
    Orquestra o pipeline de treinamento com GridSearchCV para otimizar
    cada um dos 3 modelos e salvá-los em disco.
    """
    print("--- INICIANDO SESSÃO DE TREINAMENTO OTIMIZADO COM GRIDSEARCHCV ---")

    # --- 1. Carregamento e Geração de Features (Embeddings) ---
    print("\n[ETAPA 1/4] Carregando dataset e gerando embeddings...")
    try:
        df = pd.read_csv('diarios_universitarios.csv')
    except FileNotFoundError:
        print("Erro: Arquivo 'diarios_universitarios.csv' não encontrado.")
        return
    
    embeddings = df['diary_text'].apply(gerar_embedding).dropna()
    df_processado = df.loc[embeddings.index].copy()
    X = np.vstack(embeddings.values)
    y_original = df_processado['ground_truth_label']
    print(f"Processamento de {len(df_processado)} textos concluído.")

    # --- 2. Preparação dos Alvos ---
    print("\n[ETAPA 2/4] Preparando alvos de classificação...")
    df_processado['is_anxiety'] = y_original.isin(['risco_ansiedade', 'risco_misto']).astype(int)
    df_processado['is_depression'] = y_original.isin(['risco_depressao', 'risco_misto']).astype(int)
    df_processado['is_burnout'] = y_original.isin(['risco_burnout']).astype(int)
    
    # --- 3. Otimização e Treinamento dos Modelos ---
    print("\n[ETAPA 3/4] Otimizando e treinando 3 modelos especializados...")

    # Definindo a grade de parâmetros para o GridSearchCV testar
    # É uma grade menor para ser mais rápido, mas focada nos parâmetros mais importantes.
    param_grid = {
        'n_estimators': [100, 200, 400],
        'max_depth': [10, 50, None],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10]
    }
    
    modelos_otimizados = {}
    condicoes = ['anxiety', 'depression', 'burnout']
    
    for condicao in condicoes:
        print(f"\n  - Iniciando GridSearchCV para: {condicao.upper()}...")
        y = df_processado[f'is_{condicao}']
        
        # Modelo base
        rf = RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1)
        
        # Configuração do GridSearchCV
        # cv=3 -> validação cruzada com 3 folds para acelerar
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, verbose=2, scoring='f1')
        
        # Executando a busca
        grid_search.fit(X, y)
        
        print(f"  - Busca concluída para {condicao.upper()}.")
        print(f"  - Melhores parâmetros: {grid_search.best_params_}")
        
        # Salvando o melhor modelo encontrado
        modelos_otimizados[condicao] = grid_search.best_estimator_

    print("\nOtimização e treinamento concluídos com sucesso!")

    # --- 4. Salvando os Modelos Otimizados em Disco ---
    print("\n[ETAPA 4/4] Salvando modelos otimizados na pasta 'modelos_finais'...")
    os.makedirs('modelos_finais', exist_ok=True)
    for condicao, modelo in modelos_otimizados.items():
        caminho_arquivo = os.path.join('modelos_finais', f'modelo_{condicao}.joblib')
        joblib.dump(modelo, caminho_arquivo)
        print(f"  - Modelo salvo em: {caminho_arquivo}")

    print("\n--- SESSÃO DE TREINAMENTO OTIMIZADO CONCLUÍDA COM SUCESSO ---")

if __name__ == "__main__":
    train_and_save_models()