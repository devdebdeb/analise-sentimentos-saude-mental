import pandas as pd
import joblib
import os
import numpy as np
from feature_extractor_embeddings import gerar_embedding

def predict_scores_from_csv(input_csv_path: str, output_csv_path: str):
    """
    Carrega modelos, lê um CSV de entrada, calcula os scores e salva em um CSV de saída.
    """
    print("--- INICIANDO SESSÃO DE PREDIÇÃO ---")

    # --- 1. Carregar os modelos pré-treinados ---
    print("\n[ETAPA 1/4] Carregando modelos da pasta 'modelos_finais'...")
    modelos = {}
    caminho_base = 'modelos_finais'
    condicoes = ['anxiety', 'depression', 'burnout']

    try:
        for condicao in condicoes:
            caminho_arquivo = os.path.join(caminho_base, f'modelo_{condicao}.joblib')
            modelos[condicao] = joblib.load(caminho_arquivo)
        print("Modelos carregados com sucesso!")
    except FileNotFoundError:
        print(f"Erro: Modelos não encontrados na pasta '{caminho_base}'.")
        print("Por favor, execute o script 'train_models.py' primeiro.")
        return

    # --- 2. Carregar e processar o CSV de entrada ---
    print(f"\n[ETAPA 2/4] Lendo e processando o arquivo '{input_csv_path}'...")
    try:
        df_textos = pd.read_csv(input_csv_path)
        # Remove linhas onde o texto está vazio
        df_textos.dropna(subset=['texto'], inplace=True)
        if df_textos.empty:
            print("Nenhum texto válido encontrado no arquivo de entrada.")
            return
    except FileNotFoundError:
        print(f"Erro: Arquivo de entrada '{input_csv_path}' não encontrado.")
        return
        
    embeddings = df_textos['texto'].apply(gerar_embedding).dropna()
    df_processado = df_textos.loc[embeddings.index].copy()
    X = np.vstack(embeddings.values)
    print(f"Processamento de {len(df_processado)} textos concluído.")

    # --- 3. Calcular os Scores de Risco ---
    print("\n[ETAPA 3/4] Calculando os scores de risco...")
    df_processado['score_ansiedade'] = modelos['anxiety'].predict_proba(X)[:, 1]
    df_processado['score_depressao'] = modelos['depression'].predict_proba(X)[:, 1]
    df_processado['score_burnout'] = modelos['burnout'].predict_proba(X)[:, 1]
    print("Scores calculados.")

    # --- 4. Salvar os Resultados ---
    print(f"\n[ETAPA 4/4] Salvando resultados em '{output_csv_path}'...")
    colunas_resultado = ['id_texto', 'texto', 'score_ansiedade', 'score_depressao', 'score_burnout']
    df_resultado = df_processado[colunas_resultado].round(3)
    
    df_resultado.to_csv(output_csv_path, index=False, sep=';', decimal=',')
    
    print("\n--- SESSÃO DE PREDIÇÃO CONCLUÍDA ---")
    print(f"Resultados salvos com sucesso! Verifique o arquivo '{output_csv_path}'.")
    print("\nAbaixo uma prévia dos resultados:")
    print(df_resultado)

if __name__ == "__main__":
    # Define os nomes dos arquivos de entrada e saída
    arquivo_de_entrada = '/data/textos_para_analise_novo.csv'
    arquivo_de_saida = '/data/resultados_saude_mental.csv'
    predict_scores_from_csv(input_csv_path=arquivo_de_entrada, output_csv_path=arquivo_de_saida)