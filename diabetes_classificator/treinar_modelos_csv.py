import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
import os
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, roc_auc_score, make_scorer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importações para reamostragem
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Criando diretório
os.makedirs("modelos", exist_ok=True)

# 1. Carregar o dataset
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# 2. Separar variáveis preditoras e alvo
X = df.drop(columns=['Diabetes_binary'])
y = df['Diabetes_binary']

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

#normalização min-max dos dados(não binários e dos que possui ideia de ordem)
columns_to_normalize = ["BMI","GenHlth","MentHlth","PhysHlth","Age","Education","Income"]
columns_to_pass_through = ["HighBP","HighChol","CholCheck","Stroke","HeartDiseaseorAttack","PhysActivity","Fruits","Veggies","HvyAlcoholConsump","AnyHealthcare","NoDocbcCost", "DiffWalk","Sex"]

preprocessor = ColumnTransformer(
    transformers=[
        ('minmax_scaler', MinMaxScaler(), columns_to_normalize),
        ('passthrough_features', 'passthrough', columns_to_pass_through)
    ],
    )

#balanciar o dados de treino

best_choice = ImbPipeline([
    ('preprocessor', preprocessor),
    ('under_sampler', RandomUnderSampler(random_state=42, sampling_strategy = 0.9)),
    ('over_sampler', SMOTE(random_state=42, k_neighbors = 7)),
    ('classifier', DecisionTreeClassifier(random_state=42, criterion = 'entropy', max_depth = 10))
])


# Treinar árvore de decisão

model_dt = best_choice.fit(X_train, y_train)

#avaliar modelo
y_pred = model_dt.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"F1-Score (macro): {f1_score(y_test, y_pred, average='macro'):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.4f}")


# Salvando
joblib.dump(model_dt, "diabetes_classificator/modelos/arvore_diabetes_binario.pkl")


print("✅ Modelos treinados e salvos em ./modelos")
