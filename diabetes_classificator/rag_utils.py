import streamlit as st
import os
import faiss
import getpass
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model

# Importa REFERENCIAS_BIBLIOGRAFICAS_DIABETES e get_diabetes_article_paths de utils
from utils import REFERENCIAS_BIBLIOGRAFICAS_DIABETES, get_diabetes_article_paths


@st.cache_resource(show_spinner="Carregando e processando artigos científicos de diabetes...")
def setup_diabetes_explanation_rag():
    """
    Configura o LLM, Embeddings e Vector Store para o RAG de explicações de diabetes.
    Esta função será executada apenas uma vez.
    """
    if "GROQ_API_KEY" not in os.environ:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        else:
            st.error(
                "Chave de API GROQ_API_KEY não encontrada nas variáveis de ambiente ou Streamlit secrets."
            )
            st.stop()

    _llm = init_chat_model("llama3-70b-8192", model_provider="groq")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        _embedding_dim = len(embeddings.embed_query("hello world"))
    except Exception as e:
        st.error(f"Erro ao obter dimensão do embedding do Hugging Face. Verifique a instalação da 'sentence-transformers': {e}")
        st.stop()

    _index = faiss.IndexFlatL2(_embedding_dim)
    _vector_store = FAISS(
        embedding_function=embeddings,
        index=_index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    diabetes_article_paths = get_diabetes_article_paths()

    if not diabetes_article_paths:
        st.warning(
            "Nenhum arquivo de artigo de diabetes foi carregado para o RAG de explicações. As explicações serão genéricas."
        )
        return _llm, None

    docs = []
    for path in diabetes_article_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            loader = TextLoader(path)
        docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        add_start_index=True,
    )
    all_splits = text_splitter.split_documents(docs)
    _vector_store.add_documents(documents=all_splits)

    return _llm, _vector_store


@st.cache_data(show_spinner=False)
def get_explanation_from_rag(
    feature_name,
    user_value,
    is_less_or_equal_to_threshold,
    threshold,
    _llm_rag,
    _vector_store_rag,
):
    """
    Gera uma explicação baseada em RAG para uma feature específica.
    Retorna o conteúdo da explicação e uma lista das chaves de referência citadas.
    """
    default_explanation_map = {
        "BMI": "Body Mass Index (BMI) is a common indicator of obesity, which is strongly associated with an increased risk of type 2 diabetes. Maintaining a healthy BMI is crucial for prevention.",
        "Age": "Age is a non-modifiable risk factor for type 2 diabetes, with the risk progressively increasing with aging due to metabolic changes and lower insulin sensitivity.",
        "HighBP": "High blood pressure (hypertension) is a frequent comorbidity in individuals with diabetes, and both problems share risk factors, increasing overall cardiovascular risk.",
        "HighChol": "Elevated cholesterol levels (dyslipidemia) are often observed in patients with diabetes, contributing to the risk of cardiovascular diseases, a common complication of diabetes.",
        "Smoker": "Smoking increases the risk of developing type 2 diabetes and its complications, as it can lead to insulin resistance and systemic inflammation.",
        "PhysActivity": "Regular physical activity improves insulin sensitivity and helps with weight control, significantly reducing the risk of developing type 2 diabetes. Physical inactivity, on the other hand, increases the risk.",
        "HvyAlcoholConsump": "Heavy alcohol consumption can negatively impact glucose metabolism and contribute to weight gain, increasing the risk of type 2 diabetes.",
        "GenHlth": "Self-rated general health reflects perceived well-being and the presence of other chronic conditions that can influence the risk of diabetes.",
        "MentHlth": "Mental health problems such as chronic stress, depression, and anxiety can indirectly increase the risk of diabetes due to lifestyle changes, eating habits, and hormonal responses (like cortisol).",
        "PhysHlth": "Frequent or chronic physical health problems may indicate underlying conditions or a lifestyle that increases vulnerability to diabetes.",
        "DiffWalk": "Difficulties walking or moving can be a sign of physical inactivity and other comorbidities, both of which elevate the risk of diabetes.",
        "Sex": "There are differences in the prevalence and risk profile of diabetes between biological sexes, influenced by hormonal, genetic, and lifestyle factors.",
        "Education": "Lower levels of education may be associated with less access to health information and less healthy lifestyles, impacting diabetes risk.",
        "Income": "Income is a socioeconomic factor that affects access to healthy foods, preventive medical care, and other health resources, influencing diabetes risk.",
        "Fruits": "Adequate fruit consumption, as part of a balanced diet, is important for general health and can contribute to diabetes prevention due to its fiber and nutrient content.",
        "Veggies": "Vegetable consumption is fundamental for a healthy diet, providing fiber and antioxidants that support metabolic health and can reduce the risk of diabetes.",
        "CholCheck": "Regular cholesterol checks are an indicator of health monitoring and early detection of risk factors that may be linked to diabetes.",
        "AnyHealthcare": "Having access to some form of health plan or medical coverage facilitates preventive care, early diagnosis, and management of conditions that can lead to diabetes.",
        "NoDocbcCost": "Forgoing medical visits due to cost can result in delayed diagnosis and inadequate management of health conditions, increasing the risk of diabetes progression.",
    }

    if _vector_store_rag is None:
        return default_explanation_map.get(feature_name, f"Desculpe, não há materiais de referência carregados para explicar a relação entre '{feature_name}' e o risco de diabetes. Verifique a pasta 'diabetes_articles'."), []

    general_diabetes_terms = [
        "type 2 diabetes", "DM", "prediabetes", "diabetes risk",
        "insulin resistance", "glucose metabolism", "glycemic control",
        "diabetes complications", "metabolic health", "diabetes prevention",
        "diabetes risk factors", "diabetes progression"
    ]

    query_base_terms = [f"{feature_name}"]
    query_specific_terms = []

    # Lógica de refinamento para cada feature
    # (Mantenha a lógica de refinamento da query aqui, como no código original)
    if feature_name == "BMI":
        query_specific_terms.extend(["Body Mass Index", "body weight", "obesity", "overweight",
                                     "adipocytes", "body fat", "inflammation", "adiposity"])
    if user_value <= 18.5:
        query_specific_terms.append("underweight")
    elif user_value <= 24.9:
        query_specific_terms.append("normal weight")
    elif user_value <= 29.9:
        query_specific_terms.append("overweight")
    elif user_value > 29.9:
        query_specific_terms.append("obesity")
        query_specific_terms.append("abdominal obesity")

    elif feature_name == "Age":
        query_specific_terms.extend(["age", "aging", "chronological age", "biological age",
                                     "age-related risk", "age group"])
        if user_value <= 3:
            query_specific_terms.append("young")
            query_specific_terms.append("young adults")
        elif user_value <= 9:
            query_specific_terms.append("middle-aged")
        else:
            query_specific_terms.append("elderly")
            query_specific_terms.append("seniors")
            query_specific_terms.append("elderly population")

    elif feature_name == "HighBP":
        query_specific_terms.extend(["hypertension", "high blood pressure", "cardiovascular disease",
                                     "heart disease", "stroke", "vascular complications",
                                     "endothelial dysfunction", "comorbidity"])
        if user_value == 1:
            query_specific_terms.append("with hypertension")
        else:
            query_specific_terms.append("without hypertension")

    elif feature_name == "HighChol":
        query_specific_terms.extend(["high cholesterol", "dyslipidemia", "triglycerides", "HDL-C",
                                     "LDL-C", "blood lipids", "atherosclerosis",
                                     "cardiovascular health", "lipid profile"])
        if user_value == 1:
            query_specific_terms.append("with high cholesterol")
        else:
            query_specific_terms.append("normal cholesterol")

    elif feature_name == "Smoker":
        query_specific_terms.extend(["smoking", "to smoke", "tobacco", "cigarette", "smokeless tobacco",
                                      "nicotine", "systemic inflammation", "oxidative stress",
                                      "beta-cell dysfunction", "comorbidity", "ex-smoker", "non-smoker"])
        if user_value == 1:
            query_specific_terms.append("smoker")
        else:
            query_specific_terms.append("non-smoker")

    elif feature_name == "PhysActivity":
        query_specific_terms.extend(["physical activity", "exercise", "sedentary lifestyle", "inactivity",
                                     "lifestyle", "insulin sensitivity", "weight control",
                                     "energy expenditure"])
        if user_value == 1:
            query_specific_terms.append("active")
        else:
            query_specific_terms.append("sedentary")

    elif feature_name == "HvyAlcoholConsump":
        query_specific_terms.extend(["alcohol consumption", "alcohol", "alcohol abuse", "abdominal obesity",
                                     "hepatic metabolism", "insulin sensitivity", "liver disease",
                                     "heavy consumption", "intoxication"])
        if user_value == 1:
            query_specific_terms.append("heavy alcohol consumption")
        else:
            query_specific_terms.append("moderate alcohol consumption")

    elif feature_name == "GenHlth":
        query_specific_terms.extend(["self-rated health", "perceived health", "general well-being state",
                                      "comorbidities", "chronic inflammation", "lifestyle", "quality of life",
                                      "underlying health conditions", "health perception"])
        if user_value <= 2:
            query_specific_terms.append("excellent health")
        elif user_value <= 3:
            query_specific_terms.append("good health")
        else:
            query_specific_terms.append("fair health")
            query_specific_terms.append("poor health")

    elif feature_name in ["MentHlth", "PhysHlth"]:
        if feature_name == "MentHlth":
            query_specific_terms.extend(["mental health", "chronic stress", "depression", "anxiety", "cortisol", "psychological disorders"])
        else: # PhysHlth
            query_specific_terms.extend(["physical health", "chronic conditions", "pain", "systemic inflammation", "physical limitation", "chronic illness"])
        query_specific_terms.extend(["well-being", "quality of life", "impact on metabolism", "lifestyle"])
        if user_value <= 5:
            query_specific_terms.append("few days of problems")
        else:
            query_specific_terms.append("chronic problems")
            query_specific_terms.append("frequent malaise")

    elif feature_name == "DiffWalk":
        query_specific_terms.extend(["difficulty walking", "reduced mobility", "physical activity",
                                      "sedentary lifestyle", "locomotion", "motor limitation", "physical disability"])
        if user_value == 1:
            query_specific_terms.append("with difficulty walking")
        else:
            query_specific_terms.append("without difficulty walking")

    elif feature_name == "Sex":
        query_specific_terms.extend(["biological sex", "gender", "gender differences", "risk profile", "hormones", "gender prevalence"])
        if user_value == 1: # Masculino
            query_specific_terms.append("male")
            query_specific_terms.append("men")
        else: # Feminino
            query_specific_terms.append("female")
            query_specific_terms.append("women")

    elif feature_name == "Education":
        query_specific_terms.extend(["educational level", "schooling", "socioeconomic factors",
                                      "access to health information", "healthy habits",
                                      "awareness", "health inequalities"])
        if user_value <= 2:
            query_specific_terms.append("low education")
        elif user_value >= 5:
            query_specific_terms.append("high education")
        else:
            query_specific_terms.append("intermediate education")

    elif feature_name == "Income":
        query_specific_terms.extend(["income", "socioeconomic status", "access to healthy food",
                                      "medical care", "health resources", "health inequalities"])
        if user_value <= 3:
            query_specific_terms.append("low income")
        elif user_value >= 6:
            query_specific_terms.append("high income")
        else:
            query_specific_terms.append("intermediate income")

    elif feature_name == "Fruits":
        query_specific_terms.extend(["fruit consumption", "diet", "eating patterns", "fiber",
                                      "vitamins", "metabolic health", "glycemic control",
                                      "nutrition", "healthy foods"])
        if user_value == 0:
            query_specific_terms.append("low fruit consumption")
            query_specific_terms.append("deficient diet")
        else:
            query_specific_terms.append("regular fruit consumption")

    elif feature_name == "Veggies":
        query_specific_terms.extend(["vegetable consumption", "diet", "eating patterns", "fiber",
                                      "antioxidants", "metabolic health", "healthy foods", "nutrition"])
        if user_value == 0:
            query_specific_terms.append("low vegetable consumption")
            query_specific_terms.append("inadequate diet")
        else:
            query_specific_terms.append("regular vegetable consumption")

    elif feature_name == "CholCheck":
        query_specific_terms.extend(["cholesterol check", "screening", "medical follow-up",
                                      "early detection", "health monitoring", "diagnosis"])
        if user_value == 0:
            query_specific_terms.append("did not have a cholesterol check")
            query_specific_terms.append("less medical follow-up")
        else:
            query_specific_terms.append("had a cholesterol check")
            query_specific_terms.append("active health monitoring")

    elif feature_name == "AnyHealthcare":
        query_specific_terms.extend(["access to health plan", "medical coverage", "access to health services",
                                      "preventive care", "early diagnosis", "disease management", "healthcare system"])
        if user_value == 0:
            query_specific_terms.append("no access to health plan")
            query_specific_terms.append("barriers to care")
        else:
            query_specific_terms.append("with access to health plan")

    elif feature_name == "NoDocbcCost":
        query_specific_terms.extend(["cost barriers to medical care", "access to healthcare",
                                      "socioeconomic factor", "cost of healthcare", "preventive interventions",
                                      "disease progression", "health inequality"])
        if user_value == 1:
            query_specific_terms.append("did not go to the doctor due to cost")
            query_specific_terms.append("avoided medical care")
        else:
            query_specific_terms.append("did not skip doctor due to cost")

    combined_terms = set(query_base_terms + query_specific_terms + general_diabetes_terms)
    query = f"relação entre {feature_name} e diabetes. Termos adicionais: {', '.join(combined_terms)}"

    retrieved_docs = _vector_store_rag.similarity_search(query, k=5)

    context_str = ""
    cited_references_keys = set()

    for i, doc in enumerate(retrieved_docs):
        source_path = doc.metadata.get("source", "N/A")
        source_name = os.path.basename(source_path)
        page_info = doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))

        context_str += (
            f"### Documento {i+1} (Fonte: {source_name} - Página: {page_info})\n{doc.page_content}\n\n"
        )

        for ref_key, ref_full_text in REFERENCIAS_BIBLIOGRAFICAS_DIABETES.items():
            if ref_key.replace("_", "-").lower() in source_name.replace("_", "-").lower():
                cited_references_keys.add(ref_key)
                break

    if not context_str:
        return default_explanation_map.get(feature_name, f"Desculpe, com base nos materiais de referência disponíveis, não consigo fornecer uma explicação detalhada para a relação entre '{feature_name}' e o risco de diabetes neste contexto específico. Não foram encontrados documentos relevantes para a sua consulta."), []

    prompt_template = f"""
    You are a health and diabetes specialist. Your task is to analyze how a specific health-related attribute — or its combination with other attributes — is associated with diabetes risk, using ONLY the information available in the reference documents below.

    ---

    **Target Attribute:** {feature_name}

    **Reference Documents for Consultation:**
    {context_str}

    ---

    Based EXCLUSIVELY on the information found in the documents above:

    - Provide a clear and concise explanation (2–5 sentences) of how '{feature_name}' is related to the risk of developing diabetes.
    - When applicable, consider different levels or categories of this attribute (e.g., "high" vs. "low", "presence" vs. "absence", or ranges such as "obese", "poor general health", etc.).
    - You may also correlate '{feature_name}' with other related features if doing so helps explain diabetes risk more clearly (e.g., linking 'HighBP' and 'BMI', or 'Income' and 'Education').
    - Interpret numeric scales using relevant health classifications (e.g., BMI ≥ 30 = "obesity", GenHlth = 5 = "poor general health") when supported by the documents.
    - Your explanation must be directly grounded in the contents of the provided documents.
    - For every factual claim or connection, cite the source using the format '<source_key>', where <source_key> is the document name (excluding ".pdf").

    ---

    If the reference documents do not contain sufficient information to explain this attribute's relationship with diabetes risk, you should not attempt to guess or fabricate an answer. In that case, return the default explanation for '{feature_name}' provided by the system's internal knowledge base (default_explanation_map).
    """



    try:
        response = _llm_rag.invoke(prompt_template)
        return response.content, list(cited_references_keys)
    except Exception as e:
        return (
            f"Desculpe, não foi possível gerar uma explicação detalhada neste momento devido a um erro na geração. [Erro: {e}]",
            [],
        )