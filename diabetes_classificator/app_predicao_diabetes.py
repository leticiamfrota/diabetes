import streamlit as st
import pandas as pd
import joblib
import os
import asyncio
import nest_asyncio

# Importações de funções de outros arquivos
from utils import (
    get_diabetes_article_paths,
    REFERENCIAS_BIBLIOGRAFICAS_DIABETES,
    interpretar_threshold,
)
from rag_utils import setup_diabetes_explanation_rag, get_explanation_from_rag
from model_evaluation import show_model_details_page # Importa a função de avaliação

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


# --- Função para explicar o passo da árvore com base em RAG ---
def explicar_passo_da_arvore_com_referencia(
    nome, valor_usuario, e_menor_ou_igual, threshold, _llm_rag, _vector_store_rag
):
    """
    Gera uma explicação detalhada para cada passo da árvore de decisão usando o RAG.
    Retorna a explicação formatada e uma lista das chaves de referência citadas.
    """
    with st.spinner(f"Gerando explicação para {nome} com base em artigos científicos..."):
        explanation_text, cited_keys = get_explanation_from_rag(
            nome,
            valor_usuario,
            e_menor_ou_igual,
            threshold,
            _llm_rag,
            _vector_store_rag,
        )

    interpretacao_th_formatada = interpretar_threshold(nome, threshold)
    direcao_texto_padrao = "é menor ou igual a" if e_menor_ou_igual else "é maior que"

    full_explanation = (
        f"** Decisão sobre `{nome}`:**\n"
        f"- *Regra do Modelo:* O modelo verifica se `{nome}` {direcao_texto_padrao} `{interpretacao_th_formatada}`.\n"
        f"- *Seu Valor:* `{valor_usuario}`. Como seu valor {direcao_texto_padrao} o limite, o modelo considerou:\n"
        f"- *Justificativa:* {explanation_text}"
    )
    return full_explanation, cited_keys


# ==============================================================================
# PÁGINA 1: ANÁLISE DE RISCO (PREDIÇÃO)
# ==============================================================================
def show_prediction_page():
    st.title("🩺 Análise de Risco de Diabetes com Árvore de Decisão Explicável")

    # Setup do RAG para explicações de diabetes
    _llm_rag_diabetes, _vector_store_rag_diabetes = setup_diabetes_explanation_rag()

    # Definir as listas de colunas uma vez, de forma consistente
    columns_to_normalize = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
    columns_to_pass_through = [
        "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
        "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
        "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
    ]
    feature_names = columns_to_normalize + columns_to_pass_through

    # Carregar o modelo pré-treinado
    try:
        modelo_path = "diabetes_classificator/modelos/arvore_diabetes_binario.pkl"
        modelo = joblib.load(modelo_path)
    except FileNotFoundError:
        st.error(f"Erro: O arquivo do modelo '{modelo_path}' não foi encontrado.")
        st.info("Por favor, vá para a página 'Detalhes do Modelo' para treinar e salvar um modelo primeiro.")
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
                "Qual seu peso (Km)?", min_value=0.0, max_value=400.0, value=60.0, step=0.1
            )
            Age = st.slider(
                "Qual sua faixa etária?",
                1,
                13,
                8,
                help="1: 18-24, 2: 25-29, ..., 8: 55-59, ..., 13: 80 anos ou mais",
            )
            BMI = Age/(Weight**2)
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
                    f"**Diagnóstico Preditivo: RISCO BAIXO de ter diabetes.** (Probabilidade: {probabilidade[0]*100:.1f}%)",
                    icon="✅",
                )

            with st.expander("Clique aqui para ver a explicação detalhada do resultado", expanded=True):
                st.info(
                    "A Árvore de Decisão toma uma série de decisões 'sim/não' para classificar um paciente. Abaixo está o caminho exato que o modelo seguiu com os seus dados.",
                    icon="🗺️",
                )

                from sklearn.pipeline import Pipeline # Importa aqui para evitar circular dependency se for movido para outro lugar
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

                    explanation_text, cited_keys_for_step = (
                        explicar_passo_da_arvore_com_referencia(
                            feature_name,
                            valor_usuario_feature,
                            e_menor_ou_igual,
                            threshold,
                            _llm_rag_diabetes,
                            _vector_store_rag_diabetes,
                        )
                    )
                    st.markdown(f"##### Passo {i+1}:")
                    st.markdown(explanation_text)
                    st.markdown("---")
                    all_cited_keys.update(
                        cited_keys_for_step
                    )

                st.subheader("📚 Referências Bibliográficas Completas:")
                if all_cited_keys:
                    for key in sorted(
                        list(all_cited_keys)
                    ):
                        if key in REFERENCIAS_BIBLIOGRAFICAS_DIABETES:
                            st.markdown(f"**[{key}]** {REFERENCIAS_BIBLIOGRAFICAS_DIABETES[key]}")
                        else:
                            st.markdown(f"**[{key}]** Referência detalhada não encontrada para esta chave.")
                else:
                    st.info(
                        "Nenhuma referência específica foi citada para os passos da árvore de decisão, ou os documentos não forneceram citações claras."
                    )


# ==============================================================================
# NAVEGAÇÃO PRINCIPAL
# ==============================================================================
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma página:", ["Análise de Risco (Predição)", "Detalhes do Modelo (Avaliação)"])

if page == "Análise de Risco (Predição)":
    show_prediction_page()
else:
    show_model_details_page()