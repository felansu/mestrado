# 1) Obtenha o modelo utilizando os 5 primeiros exemplos da base de dados e também os 5 últimos.
# 2) Calcule e apresente o erro quadrático médio aplicando o modelo de regressão nos 10 exemplos utilizados para obter o modelo de regressão.
# 3) Calcule e apresente o erro quadrático médio do modelo de regressão obtido nos demais exemplos.
# 4) Argumente se o modelo tem ou não uma boa capacidade de predição em novos exemplos.
# 5) Compare com os resultados do exercício anterior e argumente as possíveis diferenças de resultados.

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
    df.plot.scatter(x='Idade',
                    y='ProbAtaqueCardiaco',
                    color='DarkGreen', label='Group 2')

def estimateModel():
    # Permitimos acesso a variável global
    global df

    # Criamos duas listas, uma com: os 5 primeiros e 5 ultimos, e outra com os restantes
    df5Pe5U = df.iloc[:5,:].append(df.tail(5)).copy()
    dfRestantes = df.iloc[5:15,:].copy()

    # Estimamos o XY e XX
    # Precisamos deles para poder obter o modelo
    estimateXY(df5Pe5U)
    estimateXX(df5Pe5U)
    somatorioXX = df5Pe5U['X*X'].sum()
    somatorioXY = df5Pe5U['X*Y'].sum()

    # Agora conseguimos obter a media, b0 e b1
    mediaX = sum(df5Pe5U['Idade']) / float(len(df5Pe5U['Idade']))
    mediaY = sum(df5Pe5U['ProbAtaqueCardiaco']) / float(len(df5Pe5U['ProbAtaqueCardiaco']))
    b1 = somatorioXY / somatorioXX
    b0 = mediaY - b1 * mediaX

    df5Pe5U['modelo'] = b0 + b1 * df5Pe5U['Idade']
    dfRestantes['modelo'] = b0 + b1 * dfRestantes['Idade']

    df5Pe5U['erro'] = df5Pe5U['ProbAtaqueCardiaco'] - df5Pe5U['modelo']
    dfRestantes['erro'] = dfRestantes['ProbAtaqueCardiaco'] - dfRestantes['modelo']

    eqmDf5Pe5U = df5Pe5U['erro'].sum() ** 2 / 10
    eqmRestantes = dfRestantes['erro'].sum() ** 2 / 10

    summary(df5Pe5U, somatorioXY, somatorioXX, b0, b1, eqmDf5Pe5U)
    summaryDetail(dfRestantes)
    summaryDetail("EQM 10 últimos valores: "+str(eqmRestantes))
    plotarDados(df5Pe5U)
    plotarDados(dfRestantes)

estimateModel()
plt.show()