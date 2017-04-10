# 1) Obtenha o modelo utilizando os 10 primeiros exemplos da base de dados.
# 2) Calcule e apresente o erro quadrático médio aplicando o modelo de regressão nos mesmos 10 primeiros exemplos da base de dados.
# 3) Calcule e apresente o erro quadrático médio do modelo de regressão obtido nos demais exemplos.
# 4) Argumente se o modelo tem ou não uma boa capacidade de predição em novos exemplos.

# Formulas
# -------------------------------
# Y = modelo
# Y = B0+B1 * X + E
# B1 = SOMA(XY)/SOMA(XX)
# B0 = MEDIA(Y) - B1 * MEDIA (X)
# R² = (Exy)² / ExxEyy
# EQM: SOMA(Erros)² / N

import pandas as pd
import matplotlib.pyplot as plt

def formatColumns(df):
    df['ProbAtaqueCardiaco'] = df['ProbAtaqueCardiaco'].str.replace(',', '.').astype('str').astype(float)
    return df

def readFile(fileName):
    return formatColumns(pd.read_csv(fileName, sep=";"))

df = readFile('csv/risco_ataque_cardiaco.csv')

def estimateXY(df):
    df['X*Y'] = df['Idade'] * df['ProbAtaqueCardiaco']
    return df

def estimateXX(df):
    df['X*X'] = df['Idade'] ** 2
    return df

def summaryHeader(somatorioXY, somatorioXX, b0, b1, eqm):
    print("""
            * Somatório de X*X:     %(somatorioXX)s 
            * Somatório de X*Y:     %(somatorioXY)s
            * B0:                   %(b0)s
            * B1:                   %(b1)s
            * EQM:                  %(eqm)s
            """ % {'somatorioXX': somatorioXX, 'somatorioXY': somatorioXY, 'b0': b0, 'b1': b1, 'eqm': eqm})

def summaryDetail(df):
    print(str(df) + "\n")

def summary(df, somatorioXY, somatorioXX, b0, b1, eqm):
    summaryHeader(somatorioXY, somatorioXX, b0, b1, eqm)
    summaryDetail(df)

def plotarDados(df):
    df.plot.scatter(x='Idade', y='ProbAtaqueCardiaco')

def estimateModel():
    # Permitimos acesso a variável global
    global df

    # Criamos duas listas, uma com os 10 primeiros e outra com os 10 ultimos
    df10Primeiros = df.iloc[:-10].copy()
    df10Ultimos = df.tail(10).copy()

    # Estimamos o XY e XX para os 10 primeiros
    # Precisamos deles para poder obter o modelo
    estimateXY(df10Primeiros)
    estimateXX(df10Primeiros)
    somatorioXX = df10Primeiros['X*X'].sum()
    somatorioXY = df10Primeiros['X*Y'].sum()

    # Agora conseguimos obter a media, b0 e b1
    mediaX = sum(df10Primeiros['Idade']) / float(len(df10Primeiros['Idade']))
    mediaY = sum(df10Primeiros['ProbAtaqueCardiaco']) / float(len(df10Primeiros['ProbAtaqueCardiaco']))
    b1 = somatorioXY / somatorioXX
    b0 = mediaY - b1 * mediaX

    df10Primeiros['modelo'] = b0 + b1 * df10Primeiros['Idade']
    df10Ultimos['modelo'] = b0 + b1 * df10Ultimos['Idade']

    df10Primeiros['erro'] = df10Primeiros['ProbAtaqueCardiaco'] - df10Primeiros['modelo']
    df10Ultimos['erro'] = df10Ultimos['ProbAtaqueCardiaco'] - df10Ultimos['modelo']

    eqm10Primeiros = (df10Primeiros['erro'].sum() ** 2) / 10
    eqm10Ultimos = (df10Ultimos['erro'].sum() ** 2) / 10

    summary(df10Primeiros, somatorioXY, somatorioXX, b0, b1, eqm10Primeiros)
    summaryDetail(df10Ultimos)
    summaryDetail("EQM 10 últimos valores: "+str(eqm10Ultimos))
    plotarDados(df10Primeiros)
    plotarDados(df10Ultimos)

estimateModel()
plt.show()