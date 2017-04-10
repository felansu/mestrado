# 1) Qual o coeficiente de correlação entre cada uma das variáveis com o preço de apartamento?
# 2) Qual a variável mais importante para explicar o preço de apartamento?
# Justifique sua resposta.

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
correlacoes['tamanho'] = numpy.corrcoef(df['preco'], df['tamanho'])[0][1]**2
correlacoes['idade'] = numpy.corrcoef(df['preco'], df['idade'])[0][1]**2
correlacoes['andar'] = numpy.corrcoef(df['preco'], df['andar'])[0][1]**2
correlacoes['quartos'] = numpy.corrcoef(df['preco'], df['quartos'])[0][1]**2
correlacoes['vagas'] = numpy.corrcoef(df['preco'], df['vagas'])[0][1]**2
print("Correlações: "+str(correlacoes))
print("Melhor correlação: " + max(correlacoes, key=correlacoes.get))
