import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

def show_model_details_page():
    st.title("🔬 Detalhes sobre a Avaliação do Modelo Pré-treinado")

    modelo_path = "diabetes_classificator/modelos/arvore_diabetes_binario.pkl"
    data_path = "diabetes_binary_health_indicators_BRFSS2015.csv"

    # 1. Carregar o modelo
    try:
        if not os.path.exists("modelos"):
            st.error("O diretório 'modelos/' não foi encontrado. Por favor, certifique-se de que o script de treinamento foi executado e salvou o modelo.")
            return

        best_model = joblib.load(modelo_path)
        st.success(f"Modelo carregado com sucesso de: `{modelo_path}`")
    except FileNotFoundError:
        st.error(f"Erro: O arquivo do modelo '{modelo_path}' não foi encontrado.")
        st.info("Por favor, execute o script de treinamento (por exemplo, `treinar_modelos_csv.py`) para treinar e salvar o modelo primeiro.")
        return
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return

    # 2. Carregar o dataset para avaliação (usando a mesma divisão de teste)
    try:
        df = pd.read_csv(data_path)
        df = df.astype(float)
        X = df.drop(columns=["Diabetes_binary"])
        y = df["Diabetes_binary"]

        # Definir as listas de colunas consistentemente com o treinamento
        columns_to_normalize = ["BMI", "GenHlth", "MentHlth", "PhysHlth", "Age", "Education", "Income"]
        columns_to_pass_through = [
            "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
            "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
            "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex"
        ]
        feature_names = columns_to_normalize + columns_to_pass_through

        # Re-dividir os dados para obter o mesmo conjunto de teste (IMPORTANTE: usar o mesmo random_state)
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.20, stratify=y, random_state=42
        )
        st.success(f"Dados de teste carregados com sucesso de: `{data_path}`")

    except FileNotFoundError:
        st.error(f"Erro: O arquivo de dados '{data_path}' não foi encontrado.")
        st.info("Por favor, certifique-se de que o arquivo CSV do dataset está presente no diretório raiz.")
        return
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar os dados: {e}")
        return

    # 3. Fazer predições e calcular métricas no conjunto de teste
    st.subheader("Realizando avaliação no conjunto de teste...")
    try:
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        roc_auc = roc_auc_score(y_test, y_proba)

        st.success("Métricas de avaliação calculadas com sucesso!")

        st.header("1. Resultados da Avaliação no Conjunto de Teste")
        st.metric(label="Acurácia Geral", value=f"{accuracy:.2%}")
        st.metric(label="Recall (Classe Positiva)", value=f"{recall:.2%}")
        st.metric(label="F1-Score (Macro)", value=f"{f1_macro:.4f}")
        st.metric(label="ROC AUC", value=f"{roc_auc:.4f}")

        col1, col2 = st.columns([1, 1.5])

        with col1:
            st.subheader("Matriz de Confusão")
            fig, ax = plt.subplots(figsize=(6, 4))
            ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, ax=ax, cmap="Blues", display_labels=["Não Diabético", "Diabético"]
            )
            plt.title("Matriz de Confusão")
            st.pyplot(fig)
            plt.close(fig)

        with col2:
            st.subheader("Relatório de Classificação")
            report_dict = classification_report(
                y_test, y_pred, target_names=["Não Diabético", "Diabético"], output_dict=True
            )
            report_df = pd.DataFrame(report_dict).transpose()
            st.dataframe(report_df)

        st.markdown("---")
        st.header("2. Importância das Características (Feature Importances)")
        st.info("Este gráfico mostra quais características o modelo considerou mais importantes para fazer a predição.")

        # Acessar o classificador dentro do pipeline
        # Importa Pipeline aqui para evitar problemas de importação circular se 'model_evaluation' não precisar de 'Pipeline' para outras funções
        from sklearn.pipeline import Pipeline
        if isinstance(best_model, Pipeline):
            classifier_step = best_model.named_steps["classifier"]
        else:
            classifier_step = best_model # Se o modelo for diretamente o DecisionTreeClassifier

        importances_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": classifier_step.feature_importances_}
        ).sort_values(by="Importance", ascending=False)

        st.dataframe(importances_df)

        fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
        sns.barplot(x="Importance", y="Feature", data=importances_df.head(15), ax=ax_imp, palette="viridis")
        plt.title("Top 15 Características Mais Importantes")
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação do modelo: {e}")
        st.warning("Verifique se o modelo foi treinado com as mesmas colunas e transformações esperadas.")