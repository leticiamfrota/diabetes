# Sistema de Análise de Risco de Diabetes Explicável

Este repositório contém uma aplicação web interativa desenvolvida com Streamlit para predição de risco de diabetes. A aplicação não só prevê o risco, mas também oferece explicações detalhadas sobre as decisões do modelo e apresenta uma avaliação completa do desempenho do modelo em termos compreensíveis para o usuário final.

## Funcionalidades

* **Predição de Risco de Diabetes**: Insira suas informações de saúde e estilo de vida para obter uma previsão do seu risco de ter diabetes ou pré-diabetes.
* **Explicação da Predição (XAI)**: Entenda cada passo que o modelo de Árvore de Decisão seguiu para chegar à sua previsão, com explicações amigáveis para cada atributo.
* **Avaliação do Modelo**: Visualize métricas de desempenho do modelo (Acurácia, Recall, F1-Score, ROC AUC), matriz de confusão e um relatório de classificação detalhado.
* **Modelo e Métricas Atualizáveis**: O aplicativo baixa o modelo de predição e as métricas de avaliação mais recentes diretamente de um repositório GitHub, garantindo que você sempre utilize a versão mais recente.

## Estrutura do Projeto

O projeto é organizado nos seguintes arquivos principais:

* `app_predicao_diabetes.py`: O arquivo principal da aplicação Streamlit. Contém a interface do usuário para entrada de dados, a lógica de predição, e a funcionalidade de explicação passo a passo do diagnóstico.
* `model_evaluation.py`: Módulo Streamlit responsável por exibir os detalhes da avaliação do modelo. Ele baixa e processa um arquivo JSON contendo as métricas de desempenho e as apresenta de forma clara e explicável.
* `utils.py`: Contém funções auxiliares, como a `interpretar_threshold`, que traduzem os limiares numéricos do modelo em descrições textuais amigáveis, e mapeamentos de nomes de atributos para termos mais compreensíveis.
* `best_model.pkl`: O modelo de Machine Learning pré-treinado (um classificador de Árvore de Decisão) que é baixado automaticamente pelo `app_predicao_diabetes.py` a partir de um URL do GitHub.
* `infos_model.json`: Um arquivo JSON contendo as métricas de avaliação e informações sobre o modelo, baixado automaticamente pelo `model_evaluation.py` a partir de um URL do GitHub.

## Como Executar Localmente

Siga os passos abaixo para colocar a aplicação para rodar em sua máquina:

### Pré-requisitos

Certifique-se de ter o Python instalado (versão 3.8 ou superior é recomendada).

### 1. Clonar o Repositório

```bash
git clone <URL_DO_SEU_REPOSITORIO>
cd <nome_do_seu_repositorio>
````

### 2\. Instalar as Dependências

Instale todas as bibliotecas necessárias usando `pip`:

```bash
pip install streamlit pandas scikit-learn matplotlib seaborn requests joblib
```

### 3\. Executar a Aplicação Streamlit

Com o ambiente ativado e as dependências instaladas, execute a aplicação principal:

```bash
streamlit run app_predicao_diabetes.py
```

Isso abrirá a aplicação em seu navegador padrão.

## Uso da Aplicação

Ao iniciar a aplicação, você verá duas páginas principais na barra lateral esquerda:

1.  **Análise de Risco (Predição)**:

      * Nesta página, você preencherá um formulário com suas informações de saúde e hábitos.
      * Ao clicar em "Analisar Risco de Diabetes", o modelo fará uma predição e indicará se o risco é Alto ou Baixo.
      * Uma seção expansível "Explicação Detalhada" mostrará o caminho de decisão que o modelo seguiu com base nos seus dados, explicando a relevância de cada atributo de forma clara.

2.  **Detalhes do Modelo (Avaliação)**:

      * Esta página exibe as métricas de desempenho do modelo, como Acurácia, Recall, F1-Score e ROC AUC.
      * Cada métrica é acompanhada por uma explicação, facilitando a compreensão do desempenho do modelo.
      * Você também encontrará a matriz de confusão e um relatório de classificação detalhado, com suas respectivas explicações.
      * As informações dos parâmetros do modelo também são apresentadas aqui.


