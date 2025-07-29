import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

from sklearn.metrics import ConfusionMatrixDisplay

def download_infos_modelo():
    """
    Faz o download das infos do modelo mais recente.
    """
    url = "https://github.com/matheusfinger/model-pipeline-brfss/raw/main/model_metrics.json" # URL para o arquivo raw JSON
    output_file = "infos_model.json"

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Verifica se houve erro na requisição
        
        with open(output_file, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        st.success(f"Arquivo de métricas baixado com sucesso: {output_file}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erro ao baixar o arquivo de métricas: {e}. Verifique sua conexão com a internet ou o URL.")
        st.stop()


def show_model_details_page():
    st.title("🔬 Detalhes sobre a Avaliação do Modelo Pré-treinado")

    with st.spinner("Baixando as informações do modelo..."):
        download_infos_modelo()

    metrics_file = "infos_model.json"

    try:
        with open(metrics_file, 'r') as f:
            model_data = json.load(f)
        st.success(f"Dados do modelo e métricas carregados com sucesso de: `{metrics_file}`")
    except FileNotFoundError:
        st.error(f"Erro: O arquivo de métricas '{metrics_file}' não foi encontrado após o download.")
        st.info("Por favor, verifique se o arquivo foi baixado corretamente.")
        return
    except json.JSONDecodeError as e:
        st.error(f"Erro ao decodificar o arquivo JSON: {e}. O arquivo pode estar corrompido ou mal formatado.")
        return
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar os dados do modelo: {e}")
        return

    # Extraindo as métricas e informações do modelo
    metrics = model_data.get("metrics", {})
    model_info = model_data.get("model_info", {})
    ano = model_data.get("ano", "N/A")

    st.header("1. Informações do Modelo")
    st.write(f"Este modelo de Machine Learning foi avaliado em **{ano}**.")
    st.write(f"O tipo de modelo utilizado é: **{model_info.get('model_name', 'Não especificado')}**.")
    
    st.subheader("Parâmetros Principais do Modelo:")
    if model_info.get('model_params'):
        st.markdown("Estes são os ajustes (parâmetros) com os quais o modelo foi configurado para funcionar:")
        for param, value in model_info['model_params'].items():
            st.markdown(f"- **`{param}`**: `{value}`")
    else:
        st.info("Parâmetros do modelo não disponíveis no arquivo JSON.")

    st.markdown("---")

    st.header("2. Resultados da Avaliação no Conjunto de Teste")
    st.markdown("Para entender quão bem nosso modelo funciona, olhamos para algumas medidas-chave:")

    accuracy = metrics.get("accuracy_score", "N/A")
    recall = metrics.get("recall_score", "N/A")
    f1_score_val = metrics.get("f1_score", "N/A")
    roc_auc = metrics.get("roc_auc_score", "N/A")
    
    col_metric, col_explanation = st.columns([1, 2])

    with col_metric:
        st.metric(label="Acurácia Geral", value=f"{accuracy:.2%}" if isinstance(accuracy, (int, float)) else accuracy)
    with col_explanation:
        st.markdown("A **Acurácia** nos diz a porcentagem total de previsões corretas que o modelo fez. Por exemplo, se a acurácia é de 73%, significa que o modelo acertou em 73% de todas as suas previsões (tanto de quem tem diabetes quanto de quem não tem).")

    st.markdown("---")
    
    with col_metric:
        st.metric(label="Recall (Sensibilidade para Diabetes)", value=f"{recall:.2%}" if isinstance(recall, (int, float)) else recall)
    with col_explanation:
        st.markdown("O **Recall** (ou Sensibilidade) é especialmente importante para nós. Ele mede a capacidade do modelo de identificar corretamente todos os casos *positivos* de diabetes. Por exemplo, um recall de 65% significa que, de todas as pessoas que *realmente têm* diabetes, o modelo conseguiu identificar 65% delas. Queremos que este valor seja alto para não deixarmos muitos casos de diabetes passarem despercebidos.")

    st.markdown("---")

    with col_metric:
        st.metric(label="F1-Score", value=f"{f1_score_val:.4f}" if isinstance(f1_score_val, (int, float)) else f1_score_val)
    with col_explanation:
        st.markdown("O **F1-Score** é um equilíbrio entre a 'precisão' (quantos dos casos que o modelo previu como diabetes realmente tinham diabetes) e o 'recall'. É uma boa medida única para ver se o modelo está bom em ambos os aspectos, especialmente quando as classes (diabetes vs. não-diabetes) não são igualmente distribuídas.")

    st.markdown("---")
    
    with col_metric:
        st.metric(label="ROC AUC", value=f"{roc_auc:.4f}" if isinstance(roc_auc, (int, float)) else roc_auc)
    with col_explanation:
        st.markdown("O **ROC AUC** (Area Under the Receiver Operating Characteristic Curve) avalia a capacidade do modelo de distinguir entre as classes. Um valor de 0.5 sugere um desempenho aleatório, enquanto um valor de 1.0 indica um modelo perfeito. Quanto mais próximo de 1, melhor o modelo separa quem tem diabetes de quem não tem.")


    col1, col2 = st.columns([1, 1.5])

    with col1:
        st.subheader("Matriz de Confusão")
        st.markdown("A **Matriz de Confusão** nos mostra detalhadamente os tipos de acertos e erros do modelo:")
        st.markdown("- **Verdadeiro Negativo (VN)**: O modelo previu que a pessoa *não tem* diabetes, e ela *realmente não tem*. (Top-left)")
        st.markdown("- **Falso Positivo (FP)**: O modelo previu que a pessoa *tem* diabetes, mas ela *não tem*. (Top-right - um 'alarme falso')")
        st.markdown("- **Falso Negativo (FN)**: O modelo previu que a pessoa *não tem* diabetes, mas ela *realmente tem*. (Bottom-left - um 'diagnóstico perdido')")
        st.markdown("- **Verdadeiro Positivo (VP)**: O modelo previu que a pessoa *tem* diabetes, e ela *realmente tem*. (Bottom-right - um acerto importante)")
    with col2:
        if "confusion_matrix" in metrics and len(metrics["confusion_matrix"]) == 2 and len(metrics["confusion_matrix"][0]) == 2:
            cm = metrics["confusion_matrix"]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=["Não Diabético", "Diabético"], yticklabels=["Não Diabético", "Diabético"])
            ax.set_xlabel('Predito')
            ax.set_ylabel('Real')
            plt.title("Matriz de Confusão")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Dados da matriz de confusão não disponíveis no arquivo JSON para exibição.")

    st.subheader("Relatório de Classificação Detalhado")
    st.markdown("Este relatório fornece uma visão mais aprofundada das métricas para cada categoria (0 = Não Diabético, 1 = Diabético):")
    st.markdown("- **Precision (Precisão)**: Dos casos que o modelo previu como positivos para uma classe, quantos estavam realmente corretos.")
    st.markdown("- **Recall (Sensibilidade)**: Dos casos que realmente pertencem a uma classe, quantos o modelo conseguiu identificar corretamente.")
    st.markdown("- **f1-score**: Uma média ponderada da Precisão e do Recall.")
    st.markdown("- **support**: O número de ocorrências reais de cada classe nos dados de teste.")
    st.markdown("- **macro avg**: Média simples das métricas por classe.")
    st.markdown("- **weighted avg**: Média das métricas por classe, ponderada pelo 'support' (número de ocorrências) de cada classe.")

    col_left_spacer, col_table, col_right_spacer = st.columns([0.5, 3, 0.5])
    
    with col_table:
        if "classification_report" in metrics and isinstance(metrics["classification_report"], str):
            
            report_str = metrics["classification_report"]
            try:
                lines = report_str.strip().split('\n')
                header_line = lines[0].strip().split()
                

                data_rows = []
                for line in lines[1:]: 
                    parts = line.strip().split()
                    if not parts:
                        continue
                    if parts[0].isdigit() or parts[0] == 'accuracy' or parts[0] == 'macro' or parts[0] == 'weighted':
                        if len(parts) == 2 and parts[0] == 'accuracy': 
                            data_rows.append([parts[0], '', '', parts[1], '']) 
                        elif len(parts) >= 4: 
                            data_rows.append(parts)
                        elif len(parts) == 3 and (parts[0] == 'macro' or parts[0] == 'weighted') : 
                             data_rows.append([f"{parts[0]} {parts[1]}", parts[2], parts[3], parts[4], parts[5]]) 
                        elif len(parts) == 5:
                            data_rows.append([f"{parts[0]} {parts[1]}", parts[2], parts[3], parts[4], parts[5]])
                
                col_names = ["Category", "precision", "recall", "f1-score", "support"]
                if not header_line: 
                    report_df = pd.DataFrame(data_rows)
                else: 
                    parsed_report = {}
                    current_key = None
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        
                        if 'precision' in line and 'recall' in line and 'f1-score' in line and 'support' in line:
                            header_labels = line.split()
                            continue 
                        
                        parts = line.split()
                        if not parts:
                            continue
                        
                        if parts[0].isdigit() and len(parts) == 5:
                            category = parts[0]
                            parsed_report[category] = {
                                'precision': float(parts[1]),
                                'recall': float(parts[2]),
                                'f1-score': float(parts[3]),
                                'support': int(parts[4])
                            }
                        elif parts[0] == 'accuracy' and len(parts) == 2:
                            parsed_report['accuracy'] = float(parts[1])
                        elif (parts[0] == 'macro' or parts[0] == 'weighted') and len(parts) == 5:
                             key_name = f"{parts[0]} {parts[1]}"
                             parsed_report[key_name] = {
                                'precision': float(parts[2]),
                                'recall': float(parts[3]),
                                'f1-score': float(parts[4]),
                                'support': int(parts[5])
                            }
                        
                    
                    report_df = pd.DataFrame.from_dict(parsed_report, orient='index')
                    
                    
                    if 'accuracy' in report_df.index:
                        accuracy_val = report_df.loc['accuracy'].iloc[0] 
                        report_df.loc['accuracy', ['precision', 'recall', 'f1-score', 'support']] = [accuracy_val, accuracy_val, accuracy_val, accuracy_val]
                    
                    st.dataframe(report_df.fillna('')) 

            except Exception as e:
                st.error(f"Erro ao processar o relatório de classificação: {e}. Exibindo como texto simples.")
                st.code(metrics["classification_report"])
        else:
            st.info("Dados do relatório de classificação não disponíveis no arquivo JSON para exibição.")



# ==============================================================================
# NAVEGAÇÃO PRINCIPAL
# ==============================================================================
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha uma página:", ["Análise de Risco (Predição)", "Detalhes do Modelo (Avaliação)"])

if page == "Análise de Risco (Predição)":
    st.write("Vá para a página de 'Análise de Risco (Predição)' para usar o modelo.")
else:
    show_model_details_page()