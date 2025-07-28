import os

# Mapeamento das referências bibliográficas (geral)
REFERENCIAS_BIBLIOGRAFICAS_DIABETES = {
    "1-s2.0-S0022316622095906-main.pdf": "s00223166_2022",
    "1-s2.0-S2667036423000018-main.pdf": "s26670364_2023",
    "3061-9186-1-PB.pdf": "pb_3061_9186",
    "applsci-14-05184.pdf": "applsci_2024",
    "bmjnph-2020-000218.pdf": "bmjnph_2020",
    "cureus-0014-00000030733.pdf": "cureus_2022",
    "dc201129.pdf": "dc_2011",
    "dgab335.pdf": "dgab335_2020",
    "EBSCO-FullText-19_06_2025.pdf": "ebsco_1906_2025",
    "EBSCO-FullText-25_06_2025.pdf": "ebsco_2506_2025_0",
    "EBSCO-FullText-25_06_2025 (1).pdf": "ebsco_2506_2025_1",
    "EBSCO-FullText-25_06_2025 (1) (1).pdf": "guo_2025",
    "EBSCO-FullText-25_06_2025 (2).pdf": "humayun_2025",
    "EBSCO-FullText-25_06_2025 (3).pdf": "rao_2025",
    "EBSCO-FullText-25_06_2025 (4).pdf": "pan_2025",
    "EBSCO-FullText-25_06_2025 (5).pdf": "ebsco_2506_2025_5",
    "EBSCO-FullText-25_06_2025 (6).pdf": "flanagan_2025",
    "hr201572.pdf": "hr_2015",
    "JDB-15-881.pdf": "jdb_15_881",
    "main.pdf": "main_article",
    "NEJMoa012512.pdf": "nejm_oa_012512",
    "nihms857333.pdf": "nihms_857333",
    "nihms-691426.pdf": "nihms_691426",
    "Obesity - 2012 - Risérus - Alcohol Intake  Insulin Resistance  and Abdominal Obesity in Elderly Men.pdf": "riserus_2012",
    "s00059-025-05323-z.pdf": "dorr_2025",
    "s00592-025-02453-y.pdf": "article_02453",
    "s00592-025-05323-z.pdf": "rathi_2025",
    "s13340-023-00642-0.pdf": "yaegashi_2023",
    "s41598-019-56014-9.pdf": "nature_2019",
    "s41598-022-25813-y.pdf": "nature_2022",
    "s41598-023-33743-6.pdf": "nature_2023",
    "s41598-024-68202-3.pdf": "nature_2024",
}

def get_diabetes_article_paths():
    """
    Retorna uma lista de caminhos para os arquivos PDF de artigos sobre diabetes.
    Certifique-se de que esses arquivos estão no diretório 'diabetes_articles'.
    """
    articles_dir = "diabetes_classificator/diabetes_articles"
    if not os.path.exists(articles_dir):
        # Em um módulo utilitário, não use st.error diretamente,
        # apenas retorne vazio ou levante uma exceção para o chamador lidar.
        return []

    pdf_files = [
    os.path.join(articles_dir, "1-s2.0-S0022316622095906-main.pdf"),
    os.path.join(articles_dir, "1-s2.0-S2667036423000018-main.pdf"),
    os.path.join(articles_dir, "3061-9186-1-PB.pdf"),
    os.path.join(articles_dir, "applsci-14-05184.pdf"),
    os.path.join(articles_dir, "bmjnph-2020-000218.pdf"),
    os.path.join(articles_dir, "cureus-0014-00000030733.pdf"),
    os.path.join(articles_dir, "dc201129.pdf"),
    os.path.join(articles_dir, "dgab335.pdf"),
    os.path.join(articles_dir, "EBSCO-FullText-19_06_2025.pdf"),
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025.pdf"),
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (1).pdf"),
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (1) (1).pdf"),  # Guo
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (2).pdf"),      # Humayun
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (3).pdf"),      # Rao
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (4).pdf"),      # Pan et al.
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (5).pdf"),
    os.path.join(articles_dir, "EBSCO-FullText-25_06_2025 (6).pdf"),      # Flanagan
    os.path.join(articles_dir, "hr201572.pdf"),
    os.path.join(articles_dir, "JDB-15-881.pdf"),
    os.path.join(articles_dir, "main.pdf"),
    os.path.join(articles_dir, "NEJMoa012512.pdf"),
    os.path.join(articles_dir, "nihms857333.pdf"),
    os.path.join(articles_dir, "nihms-691426.pdf"),
    os.path.join(
        articles_dir,
        "Obesity - 2012 - Risérus - Alcohol Intake  Insulin Resistance  and Abdominal Obesity in Elderly Men.pdf",
    ),  # Risérus
    os.path.join(articles_dir, "s00059-025-05323-z.pdf"),                 # Dörr
    os.path.join(articles_dir, "s00592-025-02453-y.pdf"),
    os.path.join(articles_dir, "s00592-025-05323-z.pdf"),                 # Rathi
    os.path.join(articles_dir, "s13340-023-00642-0.pdf"),                 # Yaegashi
    os.path.join(articles_dir, "s41598-019-56014-9.pdf"),
    os.path.join(articles_dir, "s41598-022-25813-y.pdf"),
    os.path.join(articles_dir, "s41598-023-33743-6.pdf"),
    os.path.join(articles_dir, "s41598-024-68202-3.pdf"),
]

    existing_files = [f for f in pdf_files if os.path.exists(f)]
    return existing_files

def interpretar_threshold(nome_feature, threshold):
    """
    Interpreta thresholds numéricos para descrições textuais amigáveis.
    """
    match nome_feature:
        case "GenHlth":
            if threshold <= 1.5:
                return "Saúde 'Excelente'"
            if threshold <= 2.5:
                return "Saúde 'Excelente' ou 'Muito Boa'"
            if threshold <= 3.5:
                return "Saúde 'Boa' ou melhor"
            if threshold <= 4.5:
                return "Saúde 'Regular' ou melhor"
            else:
                return "Saúde 'Ruim'"
        case "Income":
            if threshold <= 2.5:
                return "Renda até $15k"
            if threshold <= 4.5:
                return "Renda até $25k"
            if threshold <= 6.5:
                return "Renda até $50k"
            else:
                return "Renda acima de $50k"
        case "Age":
            if threshold <= 1.5:
                return "18-24 anos"
            if threshold <= 2.5:
                return "25-29 anos"
            if threshold <= 3.5:
                return "30-34 anos"
            if threshold <= 4.5:
                return "35-39 anos"
            if threshold <= 5.5:
                return "40-44 anos"
            if threshold <= 6.5:
                return "45-49 anos"
            if threshold <= 7.5:
                return "50-54 anos"
            if threshold <= 8.5:
                return "55-59 anos"
            if threshold <= 9.5:
                return "60-64 anos"
            if threshold <= 10.5:
                return "65-69 anos"
            if threshold <= 11.5:
                return "70-74 anos"
            if threshold <= 12.5:
                return "75-79 anos"
            else:
                return "80 anos ou mais"
        case "BMI":
            if threshold <= 18.5:
                return "Abaixo do peso (IMC < 18.5)"
            if threshold <= 24.9:
                return "Peso normal (IMC 18.5-24.9)"
            if threshold <= 29.9:
                return "Sobrepeso (IMC 25-29.9)"
            else:
                return "Obesidade (IMC >= 30)"
        case "MentHlth" | "PhysHlth":
            if threshold <= 0.5:
                return "Nenhum dia com problemas de saúde"
            if threshold <= 5.5:
                return "Até 5 dias com problemas"
            if threshold <= 14.5:
                return "Até 14 dias com problemas"
            else:
                return "Mais de 15 dias (problemas crônicos)"
        case _:
            if threshold == 0.5:
                return "Separação entre 'Não' (0) e 'Sim' (1)"
            else:
                return f"Threshold numérico: {threshold:.1f}"