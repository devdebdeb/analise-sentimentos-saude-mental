from sentence_transformers import SentenceTransformer

# Define o nome do modelo que vamos usar.
MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

try:
    print(f"Carregando modelo de embedding: '{MODEL_NAME}'... (O download pode levar um tempo na primeira vez)")
    # Carrega o modelo. Ele será baixado e salvo em cache na primeira execução.
    MODELO_EMBEDDING = SentenceTransformer(MODEL_NAME)
    print("Modelo de embedding carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo de embedding: {e}")
    MODELO_EMBEDDING = None

def gerar_embedding(texto: str):
    """
    Recebe um texto e retorna seu vetor de embedding como um array NumPy.
    """
    if MODELO_EMBEDDING and texto:
        # A função .encode() é o coração da biblioteca: transforma o texto em um vetor.
        embedding = MODELO_EMBEDDING.encode(texto)
        return embedding
    return None

# --- Demonstração de Uso ---
if __name__ == '__main__':
    frase1 = "Estou muito feliz hoje."
    frase2 = "Sinto uma grande alegria."
    
    embedding1 = gerar_embedding(frase1)
    embedding2 = gerar_embedding(frase2)
    
    print(f"\nA frase '{frase1}' foi transformada em um vetor de {embedding1.shape[0]} dimensões.")
    # print(embedding1) 
    
    # Podemos calcular a similaridade entre os vetores (cosseno)
    from sentence_transformers.util import cos_sim
    similaridade = cos_sim(embedding1, embedding2)
    print(f"\nA similaridade entre as duas frases é: {similaridade.item():.4f}")