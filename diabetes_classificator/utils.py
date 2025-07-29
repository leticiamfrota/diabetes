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
        case "Sex": 
            if threshold < 0.5: 
                return "Feminino (valor 0)"
            elif threshold >= 0.5:
                return "Masculino (valor 1)"
            else:
                return f"Threshold numérico: {threshold:.1f} (incomum para Sexo)"
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