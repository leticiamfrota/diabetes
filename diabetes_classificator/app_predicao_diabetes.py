import streamlit as st
import pandas as pd
import joblib
import os
import asyncio
import nest_asyncio
import requests
from sklearn.pipeline import Pipeline

def download_modelo():
    """
    Faz o download do modelo mais recente.

    Returns:
        sdp_dataset (pandas.DataFrame): DataFrame com os dados carregados.
    """
    url = "https://github.com/matheusfinger/model-pipeline-brfss/raw/main/best_model.pkl"
    output_file = "best_model.pkl"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Verifica se houve erro na requisição
        
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        st.success(f"Modelo baixado com sucesso: {output_file}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o modelo: {e}. Verifique sua conexão com a internet ou o URL.")
        st.stop()

# Importações de funções de outros arquivos
from utils import (
    interpretar_threshold
)
from model_evaluation import show_model_details_page

nest_asyncio.apply()

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA E CACHE
# ==============================================================================

st.set_page_config(page_title="Sistema de Análise de Risco de Diabetes", layout="wide")


@st.cache_data
def load_data(path):
    """Carrega o dataset e o armazena em cache para performance."""
    df = pd.read_csv(path)
    return df

default_explanation_map = {
    "BMI": "Body Mass Index (BMI) é um indicador comum de obesidade, que está fortemente associado a um risco aumentado de diabetes tipo 2. Manter um IMC saudável é crucial para a prevenção.",
    "Age": "Idade é um fator de risco não modificável para diabetes tipo 2, com o risco aumentando progressivamente com o envelhecimento devido a mudanças metabólicas e menor sensibilidade à insulina.",
    "HighBP": "Pressão Alta (hipertensão) é uma comorbidade frequente em indivíduos com diabetes, e ambos os problemas compartilham fatores de risco, aumentando o risco cardiovascular geral.",
    "HighChol": "Níveis elevados de colesterol (dislipidemia) são frequentemente observados em pacientes com diabetes, contribuindo para o risco de doenças cardiovasculares, uma complicação comum do diabetes.",
    "Smoker": "Fumar aumenta o risco de desenvolver diabetes tipo 2 e suas complicações, pois pode levar à resistência à insulina e inflamação sistêmica.",
    "PhysActivity": "Atividade física regular melhora a sensibilidade à insulina e ajuda no controle do peso, reduzindo significativamente o risco de desenvolver diabetes tipo 2. A inatividade física, por outro lado, aumenta o risco.",
    "HvyAlcoholConsump": "Consumo pesado de álcool pode impactar negativamente o metabolismo da glicose e contribuir para o ganho de peso, aumentando o risco de diabetes tipo 2.",
    "GenHlth": "Saúde Geral autoavaliada reflete o bem-estar percebido e a presença de outras condições crônicas que podem influenciar o risco de diabetes.",
    "MentHlth": "Problemas de Saúde Mental, como estresse crônico, depressão e ansiedade, podem indiretamente aumentar o risco de diabetes devido a mudanças no estilo de vida, hábitos alimentares e respostas hormonais (como o cortisol).",
    "PhysHlth": "Problemas de Saúde Física frequentes ou crônicos podem indicar condições subjacentes ou um estilo de vida que aumenta a vulnerabilidade ao diabetes.",
    "DiffWalk": "Dificuldades para Caminhar ou se mover podem ser um sinal de inatividade física e outras comorbidades, ambas elevando o risco de diabetes.",
    "Sex": "Existem diferenças no perfil de prevalência e risco de diabetes entre os sexos biológicos, influenciadas por fatores hormonais, genéticos e de estilo de vida.",
    "Education": "Níveis mais baixos de Escolaridade podem estar associados a menor acesso a informações de saúde e estilos de vida menos saudáveis, impactando o risco de diabetes.",
    "Income": "Renda é um fator socioeconômico que afeta o acesso a alimentos saudáveis, cuidados médicos preventivos e outros recursos de saúde, influenciando o risco de diabetes.",
    "Fruits": "Consumo adequado de Frutas, como parte de uma dieta equilibrada, é importante para a saúde geral e pode contribuir para a prevenção do diabetes devido ao seu teor de fibras e nutrientes.",
    "Veggies": "Consumo de Vegetais é fundamental para uma dieta saudável, fornecendo fibras e antioxidantes que apoiam a saúde metabólica e podem reduzir o risco de diabetes.",
    "CholCheck": "Verificação regular de Colesterol é um indicador de monitoramento da saúde e detecção precoce de fatores de risco que podem estar ligados ao diabetes.",
    "AnyHealthcare": "Ter acesso a algum tipo de plano de saúde ou cobertura médica facilita o cuidado preventivo, o diagnóstico precoce e o manejo de condições que podem levar ao diabetes.",
    "NoDocbcCost": "Deixar de consultar um médico por causa do Custo pode resultar em diagnóstico tardio e manejo inadequado de condições de saúde, aumentando o risco de progressão do diabetes.",
}

# Novo mapa para nomes amigáveis das variáveis
friendly_feature_names = {
    "HighBP": "Pressão Alta",
    "HighChol": "Colesterol Alto",
    "CholCheck": "Checagem de Colesterol",
    "BMI": "Índice de Massa Corporal (IMC)",
    "Smoker": "Fumante",
    "Stroke": "AVC (Derrame)",
    "HeartDiseaseorAttack": "Doença Cardíaca/Infarto",
    "PhysActivity": "Atividade Física",
    "Fruits": "Consumo de Frutas",
    "Veggies": "Consumo de Vegetais",
    "HvyAlcoholConsump": "Consumo Pesado de Álcool",
    "AnyHealthcare": "Acesso a Plano de Saúde",
    "NoDocbcCost": "Não Consultou Médico por Custo",
    "GenHlth": "Saúde Geral",
    "MentHlth": "Saúde Mental",
    "PhysHlth": "Saúde Física",
    "DiffWalk": "Dificuldade para Caminhar",
    "Sex": "Sexo Biológico",
    "Age": "Faixa Etária",
    "Education": "Escolaridade",
    "Income": "Renda",
}


def explicar_passo_da_arvore(
    nome, valor_usuario, e_menor_ou_igual, threshold
):
    """
    Gera uma explicação detalhada para cada passo da árvore de decisão usando uma resposta padrão do mapa.
    Retorna a explicação formatada e uma lista vazia para as chaves de referência citadas.
    """
    interpretacao_th_formatada = interpretar_threshold(nome, threshold)
    direcao_texto_padrao = "menor ou igual a" if e_menor_ou_igual else "maior que"

    # Obtém o nome amigável do atributo, se disponível, senão usa o nome da variável
    nome_amigavel = friendly_feature_names.get(nome, nome)

    # Get explanation from the map, or a generic one if not found
    standard_explanation = default_explanation_map.get(
        nome,
        f"Esta decisão se baseia em um limite predefinido para a característica '{nome_amigavel}'. "
        f"Seu valor de '{valor_usuario}' foi comparado com o limite de '{interpretacao_th_formatada}'. "
        f"Esta é uma das muitas regras que o modelo considera para determinar o risco de diabetes."
    )

    # Nova formatação da saída
    full_explanation = (
        f"Para o atributo **{nome_amigavel}**, seu valor é `{valor_usuario}` e o limiar é `{interpretacao_th_formatada}`. "
        f"Esse atributo é importante para o diagnóstico porque {standard_explanation}"
    )
    return full_explanation

# ==============================================================================
# PÁGINA 1: ANÁLISE DE RISCO (PREDIÇÃO)
# ==============================================================================
def show_prediction_page():
    st.title("🩺 Análise de Risco de Diabetes com Árvore de Decisão Explicável")
    with st.spinner("Verificando e baixando o modelo mais recente..."):
        download_modelo()

    columns_to_normalize = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    columns_to_pass_through = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
    ]
    feature_names = columns_to_normalize + columns_to_pass_through

    try:
        modelo_path = "best_model.pkl"
        modelo = joblib.load(modelo_path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo do modelo '{modelo_path}' não foi encontrado após o download.")
        st.info("Por favor, verifique a conexão ou tente novamente.")
        st.stop()
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        st.stop()

    def entrada_binaria(label, help_text=""):
        return 1 if st.radio(label, ["Sim", "Não"], horizontal=True, help=help_text) == "Sim" else 0

    with st.form("formulario_dados"):
        st.subheader("Por favor, preencha as informações abaixo:")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### Saúde e Hábitos")
            HighBP = entrada_binaria("Tem pressão alta?", "Diagnóstico médico de hipertensão.")
            HighChol = entrada_binaria("Tem colesterol alto?", "Diagnóstico médico de colesterol alto.")
            CholCheck = entrada_binaria("Fez exame de colesterol nos últimos 5 anos?")
            Smoker = entrada_binaria("É fumante? (pelo menos 100 cigarros na vida)")
            Stroke = entrada_binaria("Já teve um AVC (derrame)?")
            HeartDiseaseorAttack = entrada_binaria("Tem doença cardíaca ou já teve um infarto?")
            MentHlth = st.number_input(
                "Agora, pensando em sua saúde mental, que inclui estresse, depressão e problemas emocionais, por quantos dias nos últimos 30 dias sua saúde mental não foi boa? Escala de 1 a 30 dias", min_value=0.0, max_value=30.0, value=0.0, step=1.0
            )

        with col2:
            st.markdown("##### Estilo de Vida e Acesso à Saúde")
            PhysActivity = entrada_binaria(
                "Pratica atividade física regularmente?", "Atividade física no último mês, além do trabalho regular."
            )
            Fruits = entrada_binaria("Consome frutas 1 ou mais vezes por dia?")
            Veggies = entrada_binaria("Consome vegetais 1 ou mais vezes por dia?")
            HvyAlcoholConsump = entrada_binaria(
                "Consumo pesado de álcool?", "Homens: 14+ doses/semana. Mulheres: 7+ doses/semana."
            )
            AnyHealthcare = entrada_binaria("Tem algum tipo de plano de saúde?")
            NoDocbcCost = entrada_binaria("Já deixou de consultar um médico por causa do custo?")
            DiffWalk = entrada_binaria("Tem séria dificuldade de caminhar ou subir escadas?")
            PhysHlth = st.number_input(
                "Agora, pensando em sua saúde física, que inclui doenças físicas e lesões, por quantos dias nos últimos 30 dias sua saúde física não foi boa? Escala de 1 a 30 dias", min_value=0.0, max_value=30.0, value=0.0, step=1.0
            )

        with col3:
            st.markdown("##### Dados Demográficos e Gerais")
            GenHlth = st.slider("Como você avalia sua saúde em geral? (1=Excelente, 5=Ruim)", 1, 5, 3)
            Weight = st.number_input(
                "Qual seu peso (Kg)?", min_value=0.0, max_value=400.0, value=60.0, step=0.1
            )
            Height = st.number_input(
                "Qual sua Altura (metros)?", min_value=0.0, max_value=3.0, value=1.6, step=0.1
            )
            Age = st.slider(
                "Qual sua faixa etária?",
                1,
                13,
                8,
                help="1: 18-24, 2: 25-29, ..., 8: 55-59, ..., 13: 80 anos ou mais",
            )
            BMI = Weight/(Height**2)
            Income = st.slider(
                "Qual sua faixa de renda anual? (1=Menor, 8=Maior)",
                1,
                8,
                5,
                help="1: <$10k, ..., 5: <$35k, ..., 8: >$75k",
            )
            Education = st.slider(
                "Qual seu nível de escolaridade? (1=Nenhuma, 6=Superior completo)", 1, 6, 4
            )
            Sex = 1 if st.selectbox("Qual seu sexo biológico?", ["Feminino", "Masculino"]) == "Masculino" else 0


        submitted = st.form_submit_button("Analisar Risco de Diabetes")

        if submitted:
            entrada_dict = {
                "HighBP": HighBP,
                "HighChol": HighChol,
                "CholCheck": CholCheck,
                "BMI": BMI,
                "Smoker": Smoker,
                "Stroke": Stroke,
                "HeartDiseaseorAttack": HeartDiseaseorAttack,
                "PhysActivity": PhysActivity,
                "Fruits": Fruits,
                "Veggies": Veggies,
                "HvyAlcoholConsump": HvyAlcoholConsump,
                "AnyHealthcare": AnyHealthcare,
                "NoDocbcCost": NoDocbcCost,
                "GenHlth": GenHlth,
                "MentHlth": MentHlth,
                "PhysHlth": PhysHlth,
                "DiffWalk": DiffWalk,
                "Sex": Sex,
                "Age": Age,
                "Education": Education,
                "Income": Income,
            }

            entrada_usuario_df = pd.DataFrame([entrada_dict])
            entrada_usuario_df = entrada_usuario_df[
                feature_names
            ]

            predicao = modelo.predict(entrada_usuario_df)[0]
            probabilidade = modelo.predict_proba(entrada_usuario_df)[0]
            st.divider()
            st.subheader("Resultado da Análise")
            if predicao == 1:
                st.error(
                    f"**Diagnóstico Preditivo: RISCO ALTO de ter diabetes ou pré-diabetes.** (Probabilidade: {probabilidade[1]*100:.1f}%)",
                    icon="🚨",
                )
            else:
                st.success(
                    f"**Diagnóstico Preditivo: RISCO BAIXO de ter diabetes.** (Probabilidade: {probabilidade[1]*100:.1f}%)",
                    icon="✅",
                )

            with st.expander("Clique aqui para ver a explicação detalhada do resultado", expanded=True):
                st.info(
                    "A Árvore de Decisão toma uma série de decisões 'sim/não' para classificar um paciente. Abaixo está o caminho exato que o modelo seguiu com os seus dados.",
                    icon="🗺️",
                )

                if isinstance(modelo, Pipeline):
                    tree_model = modelo.named_steps["classifier"]
                    entrada_usuario_transformada = modelo.named_steps["preprocessor"].transform(
                        entrada_usuario_df
                    )
                else:
                    tree_model = modelo
                    entrada_usuario_transformada = entrada_usuario_df.to_numpy()

                decision_path = tree_model.decision_path(entrada_usuario_transformada).indices

                all_cited_keys = set()

                for i, node_index in enumerate(decision_path):
                    if i == len(decision_path) - 1:
                        resultado_no_final = tree_model.tree_.value[node_index][0]
                        total_amostras = resultado_no_final.sum()
                        perc_positivo = (resultado_no_final[1] / total_amostras) * 100
                        st.markdown(
                            f"**🏁 Resultado Final:** O caminho terminou em um grupo onde **{perc_positivo:.1f}%** dos indivíduos tinham risco de diabetes."
                        )
                        continue

                    feature_index = tree_model.tree_.feature[node_index]
                    feature_name = feature_names[feature_index]
                    threshold = tree_model.tree_.threshold[node_index]

                    valor_usuario_feature = entrada_usuario_df[feature_name].iloc[0]

                    e_menor_ou_igual = valor_usuario_feature <= threshold

                    explanation_text = (
                        explicar_passo_da_arvore(
                            feature_name,
                            valor_usuario_feature,
                            e_menor_ou_igual,
                            threshold,
                        )
                    )
                    st.markdown(f"##### Passo {i+1}:")
                    st.markdown(explanation_text)


# ==============================================================================
# NAVEGAÇÃO PRINCIPAL
# ==============================================================================
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma página:", ["Análise de Risco (Predição)", "Detalhes do Modelo (Avaliação)"])

if page == "Análise de Risco (Predição)":
    show_prediction_page()
else:
    show_model_details_page()