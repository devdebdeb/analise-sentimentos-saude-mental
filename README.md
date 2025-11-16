# Sistema Preditivo de Evasão Universitária: Integração de Saúde Mental (PLN) e Machine Learning (XGBoost)

Este projeto desenvolveu uma solução de **Inteligência Artificial (IA)** para enfrentar o desafio da evasão no ensino superior. O projeto valida a hipótese de que o **risco à saúde mental** é o fator de maior peso na predição de abandono no ensino superior, integrando Processamento de Linguagem Natural (PLN) com Machine Learning Avançado (XGBoost).

O pipeline técnico integra duas frentes:
1.  **Processamento de Linguagem Natural (PLN):** Transforma textos de diários anônimos em *scores* quantitativos de risco de Ansiedade, Depressão e Burnout.
2.  **Machine Learning Avançado:** Utiliza o algoritmo **XGBoost Otimizado** para integrar esses *scores* com dados demográficos e acadêmicos, gerando uma previsão final de evasão (`Dropout`).

---

### Disclaimer Ético

**Esta ferramenta é um protótipo experimental e não substitui, de forma alguma, um diagnóstico médico.** Os resultados são baseados em padrões estatísticos. A aplicação em um cenário real exigiria validação clínica e um protocolo de intervenção ética rigoroso, sempre com o acompanhamento de profissionais de saúde mental.

---

### Pipeline Metodológico e Funcionalidades

**Nota sobre os Dados:** Os dados estruturados (`dataset_brasil_base.csv`) e os textos de diário (`textos_para_analise_novo.csv`) são **sintéticos**, gerados através de um script Python (seguindo distribuições estatísticas e correlações lógicas). Eles simulam padrões de comportamento e saúde mental no contexto universitário brasileiro e foram cruciais para treinar e validar este protótipo.

O projeto seguiu uma metodologia de Machine Learning robusta para garantir a máxima performance preditiva na classe minoritária (`Dropout`):

1.  **Geração de Features Semânticas:** Utiliza o modelo `paraphrase-multilingual-mpnet-base-v2` (via `sentence-transformers`) para converter textos em *embeddings* e gerar os **Scores de Risco** (features NLP).
2.  **Feature Engineering Avançado:** Cria novas features de interação (ex: `impacto_depressao_notas`, `risco_burnout_comportamental`) para capturar correlações complexas antes do treinamento.
3.  **Tratamento de Desequilíbrio (SMOTE):** Aplica a técnica SMOTE no conjunto de treinamento para corrigir o desequilíbrio entre as classes `Graduate` vs. `Dropout`.
4.  **Classificação Otimizada:** Treina um modelo **XGBoost** com otimização de hiperparâmetros (`GridSearchCV`) para alcançar o pico de desempenho.
5.  **Calibração de Alerta:** Otimiza o limiar de probabilidade (`threshold`) para maximizar o **Recall** e focar na intervenção precoce.

---

### Ganhos de Performance e Validação

O processo de *Fine-Tuning* foi essencial para transformar um modelo ineficaz em um sistema de alerta funcional.

#### Comparação de Desempenho (Classe `Dropout`)

| Fase do Treinamento | Algoritmo | Correção | Recall (Captação) | Precision (Confiabilidade) | F1-Score (Equilíbrio) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline (Inicial)** | RandomForest | Nenhuma | **0.06 (6%)** | 0.42 | 0.11 |
| **2. Balanceamento** | RandomForest | SMOTE | **0.39 (39%)** | 0.36 | 0.38 |
| **3. Performance Final** | **XGBoost Otimizado** | GridSearchCV + FE | **0.43 (43%)** | **0.52** | **0.47** |

#### Evidência da Saúde Mental como Fator Dominante

A análise de importância das *features* comprova que a saúde mental é o preditor de maior peso:

| Posição | Feature | Tipo de Dado | Importância (%) |
| :--- | :--- | :--- | :--- |
| **1º** | **`autoavaliação_felicidade_Baixa`** | Saúde Mental (Estruturada) | **33.56%** |
| **2º** | **`autoavaliação_felicidade_Moderada`** | Saúde Mental (Estruturada) | **7.76%** |
| **7º** | **`notas_periodo`** | Acadêmica | 2.64% |

**Conclusão:** O fator emocional de saúde mental é o preditor de maior peso no modelo final.

#### Calibração para Intervenção (Máxima Eficácia)

Ao ajustar o limiar de decisão para **0.19** (em vez do padrão 0.50), o sistema atinge o pico de utilidade para a retenção:

* **Recall Máximo:** $\approx \mathbf{72\%}$
* **Significado:** O sistema é capaz de identificar **7 em cada 10 alunos** que evadiriam, o que é ideal para o setor de intervenção da universidade.

--- 

#### Priorização de Risco em 3 Níveis

| Nível de Prioridade        | Limiar de Probabilidade | Ação Sugerida                                    | Frequência |
|----------------------------|-------------------------|--------------------------------------------------|------------|
| **PRIORIDADE 1: CRÍTICO**  | **P ≥ 0.50**            | Foco imediato. Intervenção de alta confiança.    | 259        |
| **PRIORIDADE 2: ALTO**     | **0.19 ≤ P < 0.50**     | Monitoramento ativo. Ações programadas.          | 60         |
| **PRIORIDADE 3: PADRÃO**   | **P < 0.19**            | Monitoramento de rotina. Risco muito baixo.      | 6815       |


### Tecnologias Utilizadas

* **Python 3**
* **XGBoost:** Algoritmo final de classificação de alta performance.
* **Imbalanced-learn (imblearn):** Técnicas de balanceamento de classes (SMOTE).
* **Pandas / Scikit-learn / Joblib**
* **Sentence-Transformers:** Geração de *embeddings* de texto.

---

### Estrutura do Projeto

A estrutura foi reorganizada para seguir padrões profissionais, separando código (`src/`), dados (`data/`) e modelos binários (`modelos_finais/`).

```
|-- src/                    # CÓDIGOS FONTE PYTHON (Scripts principais)
|   |-- train_models.py                 # Treina os 3 modelos de risco NLP
|   |-- predict_on_csv.py               # Gera os scores de risco a partir dos novos textos
|   |-- train_final_classifier.py       # CLASSIFICADOR FINAL: Merge, SMOTE, XGBoost e Otimização
|   |-- analyse_features.py             # Analisa e exibe a importância das features
|   |-- feature_extractor_embeddings.py 
|   |-- predict_dropout.py              # Previsão final
|   |-- threshold_optimizer.py          # Calibração Limiar
|   |-- trainer_embeddings.py           
|-- data/                   # ARQUIVOS DE DADOS (CSV)
|   |-- dataset_brasil_base.csv         # Dataset base demográfico/acadêmico
|   |-- dataset_final_incrementado.csv  # DATASET CONSOLIDADO (Todas as colunas + Scores NLP)
|   |-- resultados_saude_mental.csv     # Scores de risco gerados pelo PLN (Intermediário)
|   |-- diarios_universitários.csv      # Dataset de textos para treinar os modelos de riscos
|   |-- texto_para_analise_novo.csv     # Um dataset de textos de exemplo ou dummy usado para testar o pipeline de inferência
|   |-- texto_para_analise.csv          # Versão alternativa ou inicial do arquivo de textos para análise.
|-- modelos_finais/         # ARQUIVOS BINÁRIOS SALVOS
|   |-- modelo_final_MAX_PERFORMANCE.joblib # O Modelo Final de Predição de Evasão (XGBoost)
|   |-- modelo_anxiety.joblib               # Modelo de classificação de risco de Ansiedade
|   |-- modelo_burnout.joblib               # Modelo de classificação de risco de Burnout
|   |-- modelo_depression.joblib            # Modelo de classificação de risco de Depressão
|   |-- modelo_final_evasao_OTIMIZADO.joblib
|   |-- modelo_final_evacao_SMOTE.joblib
|   |-- modelo_final_evasao_XGBOOST_OTIMIZADO.joblib
|   |-- modelo_fina_evasao.joblib
|   |-- scaler_final.joblib
|-- requirements.txt        # Lista de dependências
|-- README.md               # Este arquivo
```
---

### Como Usar (Fluxo de Execução Simplificado)

Para rodar o pipeline completo, você precisa primeiro treinar os modelos de PLN e, em seguida, rodar o classificador final.

#### 1. Configurar o Ambiente

```bash
# Clone o repositório e acesse o diretório
git clone [https://github.com/devdebdeb/analise-sentimentos-saude-mental.git](https://github.com/devdebdeb/analise-sentimentos-saude-mental.git)
cd analise-sentimentos-saude-mental

# Crie e ative o ambiente virtual (venv)
python -m venv .venv
# Para Windows (PowerShell/CMD):
.venv\Scripts\activate
# Para macOS/Linux (Bash/Zsh):
source .venv/bin/activate

# Instale todas as dependências do projeto
pip install -r requirements.txt
```
---

#### 2. Treinar Modelos de Risco (Geração de Base)

Este passo treina e salva os 3 modelos de Ansiedade/Depressão/Burnout.

```bash
python src/train_models.py
```

#### 3. Rodar Pipeline Completo (Treinamento Final). Treinar Modelos de Risco (Geração de Base)
```bash
# Gere os scores de risco a partir do dataset de texto sintético
python src/predict_on_csv.py 

# Execute o Classificador Final: Junta os dados, aplica SMOTE, e treina o XGBoost Otimizado
python src/train_final_classifier.py
```