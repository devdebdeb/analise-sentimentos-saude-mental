# Sistema de Avaliação de Risco à Saúde Mental com PLN

Este projeto utiliza Processamento de Linguagem Natural (PLN) e Machine Learning para analisar textos de diários anônimos e identificar padrões linguísticos associados a riscos de depressão, ansiedade e burnout em estudantes universitários.

---

### ⚠️ Disclaimer Ético

**Esta ferramenta é um protótipo experimental e não substitui, de forma alguma, um diagnóstico médico.** Os resultados são baseados em padrões estatísticos e destinam-se a demonstrar uma abordagem tecnológica. A aplicação em um cenário real exigiria validação clínica e um protocolo de intervenção ética rigoroso, sempre com o acompanhamento de profissionais de saúde mental.

---

### 🧠 Funcionalidades

O pipeline deste projeto realiza as seguintes tarefas:

1.  **Extração de Features Semânticas:** Utiliza o modelo `paraphrase-multilingual-mpnet-base-v2` (via `sentence-transformers`) para converter textos em *embeddings* (vetores de significado), capturando o contexto e a nuance da linguagem.
2.  **Treinamento de Modelos Especializados:** Treina três classificadores `RandomForestClassifier` independentes, um para cada condição de risco (Ansiedade, Depressão, Burnout), usando um dataset sintético com mais de 300 exemplos.
3.  **Predição de Scores de Risco:** Carrega os modelos treinados e os aplica a novos textos para gerar um score de probabilidade (0 a 1) para cada uma das três condições.

---

### 🛠️ Tecnologias Utilizadas

* **Python 3**
* **Pandas:** Manipulação de dados
* **Scikit-learn:** Treinamento e avaliação dos modelos de Machine Learning
* **Sentence-Transformers (Hugging Face):** Geração de embeddings de texto
* **Joblib:** Salvamento e carregamento dos modelos treinados
* **NLTK:** Ferramentas auxiliares de PLN

---

### 📂 Estrutura do Projeto

```
/
|-- modelos/                  # (Ignorado pelo Git) Pasta para modelos baixados
|-- modelos_finais/           # (Ignorado pelo Git) Pasta para modelos treinados
|-- diarios_universitarios.csv # Dataset principal para treinamento
|-- textos_para_analise.csv   # Arquivo de exemplo para novas predições
|-- feature_extractor_embeddings.py # Script que gera os embeddings
|-- train_models.py           # Script para treinar e salvar os modelos
|-- predict_on_csv.py         # Script para usar os modelos e fazer predições
|-- requirements.txt          # Lista de dependências do projeto
|-- .gitignore                # Arquivos a serem ignorados pelo Git
|-- README.md                 # Este arquivo
```

---

### 🚀 Como Usar

Siga os passos abaixo para replicar o ambiente e executar o projeto.

#### 1. Clonar e Configurar o Ambiente

```bash
# Clone este repositório
git clone [https://github.com/seu-usuario/nome-do-repositorio.git](https://github.com/seu-usuario/nome-do-repositorio.git)
cd nome-do-repositorio

# Crie e ative um ambiente virtual
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No macOS/Linux:
# source .venv/bin/activate

# Instale as dependências
pip install -r requirements.txt
```

#### 2. Baixar os Modelos de PLN

Este projeto usa modelos de linguagem que são carregados dinamicamente. Na primeira execução, eles serão baixados e salvos em cache.
* O `feature_extractor_embeddings.py` baixará o modelo `sentence-transformers`.
* *Nota: O projeto foi adaptado para também suportar modelos locais na pasta `/modelos/`, caso o download automático falhe.*

#### 3. Treinar os Modelos de Classificação

Execute o script de treinamento. Ele irá ler o `diarios_universitarios.csv`, gerar os embeddings e salvar os três modelos treinados na pasta `/modelos_finais/`.

```bash
python train_models.py
```

#### 4. Fazer Predições em Novos Dados

1.  Abra o arquivo `textos_para_analise.csv` e adicione os textos que você deseja analisar.
2.  Execute o script de predição:

```bash
python predict_on_csv.py
```

3.  Os resultados serão salvos no arquivo `resultados_analise.csv`.