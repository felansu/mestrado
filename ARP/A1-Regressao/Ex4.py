# 1) Qual é o preço previsto de um imóvel com 80m2, 10 anos e que está no 9º andar?

import pandas as pd
import numpy

def formatColumns(df):
    df['preco'] = df['preco'].str.replace(',', '.').astype('str').astype(float)
    df['tamanho'] = df['tamanho'].str.replace(',', '.').astype('str').astype(float)
    df['idade'] = df['idade'].str.replace(',', '.').astype('str').astype(float)
    df['andar'] = df['andar'].str.replace(',', '.').astype('str').astype(float)
    df['quartos'] = df['quartos'].str.replace(',', '.').astype('str').astype(float)
    df['vagas'] = df['vagas'].str.replace(',', '.').astype('str').astype(float)
    return df

def readFile(fileName):
    return formatColumns(pd.read_csv(fileName, sep=";"))

df = readFile('csv/preco_apartamentos.csv')
correlacoes = {}
correlacoes = numpy.corrcoef(df['preco'], [df['tamanho'], df['idade'],  df['andar'], df['quartos'], df['vagas']])
print("Correlações: "+str(correlacoes))
