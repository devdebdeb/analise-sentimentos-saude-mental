import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
import numpy as np

def train_evasion_model():
    print("--- INICIANDO TREINAMENTO FINAL COM FEATURE ENGINEERING + XGBOOST ---")

    # --- 1. Definir Caminhos dos Arquivos ---
    BASE_DATA_PATH = '/data/dataset_brasil_base.csv'
    SCORES_DATA_PATH = '/data/resultados_saude_mental.csv'
    FINAL_DF_PATH = '/data/dataset_final_incrementado.csv' # <-- NOVO NOME DO ARQUIVO SALVO
    MODEL_OUTPUT_PATH = '/modelos_finais/modelo_final_MAX_PERFORMANCE.joblib'
    
    if not os.path.exists(SCORES_DATA_PATH) or not os.path.exists(BASE_DATA_PATH):
        print(f"\nErro: Certifique-se de que '{BASE_DATA_PATH}' e '{SCORES_DATA_PATH}' estão no diretório.")
        return

    # --- 2. Carregar e Combinar os Datasets ---
    print("\n[ETAPA 1/7] Carregando e combinando dados...")
    
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

    # 🚨 NOVO PASSO: SALVAR O DATAFRAME COMPLETO NO DISCO 🚨
    df_final.to_csv(FINAL_DF_PATH, index=False, sep=';', decimal=',')
    print(f"Dataset final incrementado salvo como '{FINAL_DF_PATH}' (Contém todas as colunas + scores do PLN).")
    
    # --- 3. Feature Engineering Avançado ---
    print("\n[ETAPA 2/7] Criando features de interação avançada...")
    
    # Risco de Burnout
    df_final['risco_burnout_comportamental'] = (
        df_final['score_burnout'] * df_final['horas_tela_dia'] / (df_final['horas_sono'] + 0.1)
    )
    
    # Impacto da Depressão na Performance
    df_final['impacto_depressao_notas'] = df_final['score_depressao'] / (df_final['notas_periodo'] + 0.1)

    # Vulnerabilidade Acadêmica
    df_final['vulnerabilidade_academica'] = df_final['score_ansiedade'] * df_final['faltas_mensais']
    
    # --- 4. Preparação dos Dados ---
    print("\n[ETAPA 3/7] Preparando features e alvo...")
    
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
    y_encoded = y.replace({'Graduate': 0, 'Dropout': 1}) 

    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    
    # --- 5. SMOTE + Treinamento Otimizado (XGBOOST) ---
    print("\n[ETAPA 4/7] Aplicando SMOTE e Otimizando o Modelo...")
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [6, 9],
        'learning_rate': [0.05, 0.1, 0.2],
    }

    xgb_base = XGBClassifier(random_state=42, n_jobs=-1, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(
        estimator=xgb_base, 
        param_grid=param_grid, 
        cv=5, 
        verbose=0, 
        scoring='f1_macro' 
    )

    grid_search.fit(X_train_smote, y_train_smote) 
    model_evasion = grid_search.best_estimator_

    print("\n[ETAPA 5/7] Avaliando o Modelo Otimizado...")
    
    # Avaliação no conjunto de teste original
    y_pred_encoded = model_evasion.predict(X_test)
    y_pred = np.where(y_pred_encoded == 1, 'Dropout', 'Graduate')
    y_test_original = y_test.replace({0: 'Graduate', 1: 'Dropout'})
    
    print("\n--- Relatório de Classificação Final ---\n")
    print(classification_report(y_test_original, y_pred))
    print(f"Acurácia Geral: {accuracy_score(y_test_original, y_pred):.4f}")
    
    # --- 6. Salvamento Final ---
    print("\n[ETAPA 6/7] Salvando o modelo e a importância das features...")
    
    joblib.dump(model_evasion, MODEL_OUTPUT_PATH)
    print(f"✅ Modelo salvo como '{MODEL_OUTPUT_PATH}'.")
    
    # Análise de Importância das Features (Para registro)
    importances = model_evasion.feature_importances_
    feature_names = X.columns
    
    feature_importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importances.head(10).to_markdown(index=False))


if __name__ == '__main__':
    train_evasion_model()