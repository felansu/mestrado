# 1) Apresente o gráfico das funções de probabilidade: p(SL | Ci), p(SW | Ci), p(PL | Ci), p(SW |Ci)
# 2) E o gráfico do modelo de classificação p(Ci | SL, SW, PL, PW) usando o classificador Bayesiano.
# 3) Nota: as variáveis em questão são variáveis contínuas.

import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def formatColumns(df):
    return df


def readFile(fileName):
    return formatColumns(pd.read_csv(fileName, sep=","))


df = readFile('iris.data')


def pegaDadosFiltrados(df):
    return df.iloc[:35, :].append(df.iloc[50:85, :]).append(df.iloc[100:135, :]).copy()


dadosTreinamento = pegaDadosFiltrados(df)

# 1) Gaussianas
# p(SL | Ci) = Achamos as gaussianas == distribuição normal
# Gaussianas = Achar a media e desvio padrão
medias = {}
variancias = {}
desvioPadrao = {}

# 1.1
def calcularMedias(dft):
    global medias
    medias['setosa-sl'] = dft[(dft['class'] == 'Iris-setosa')]['sl'].sum() / 35
    medias['setosa-sw'] = dft[(dft['class'] == 'Iris-setosa')]['sw'].sum() / 35
    medias['setosa-pl'] = dft[(dft['class'] == 'Iris-setosa')]['pl'].sum() / 35
    medias['setosa-pw'] = dft[(dft['class'] == 'Iris-setosa')]['pw'].sum() / 35
    medias['versicolor-sl'] = dft[(dft['class'] == 'Iris-versicolor')]['sl'].sum() / 35
    medias['versicolor-sw'] = dft[(dft['class'] == 'Iris-versicolor')]['sw'].sum() / 35
    medias['versicolor-pl'] = dft[(dft['class'] == 'Iris-versicolor')]['pl'].sum() / 35
    medias['versicolor-pw'] = dft[(dft['class'] == 'Iris-versicolor')]['pw'].sum() / 35
    medias['virginica-sl'] = dft[(dft['class'] == 'Iris-virginica')]['sl'].sum() / 35
    medias['virginica-sw'] = dft[(dft['class'] == 'Iris-virginica')]['sw'].sum() / 35
    medias['virginica-pl'] = dft[(dft['class'] == 'Iris-virginica')]['pl'].sum() / 35
    medias['virginica-pw'] = dft[(dft['class'] == 'Iris-virginica')]['pw'].sum() / 35
    return medias

# 1.2 Variáncia: Valor menos a media elevado ao quadrado
def calcularVariancias(dft):
    variancias['setosa-sl'] = ((dft[(dft['class'] == 'Iris-setosa')]['sl'] - medias['setosa-sl']) ** 2).sum()/34
    variancias['setosa-sw'] = ((dft[(dft['class'] == 'Iris-setosa')]['sw'] - medias['setosa-sw']) ** 2).sum()/34
    variancias['setosa-pl'] = ((dft[(dft['class'] == 'Iris-setosa')]['pl'] - medias['setosa-pl']) ** 2).sum()/34
    variancias['setosa-pw'] = ((dft[(dft['class'] == 'Iris-setosa')]['pw'] - medias['setosa-pw']) ** 2).sum()/34
    variancias['versicolor-sl'] = ((dft[(dft['class'] == 'Iris-versicolor')]['sl'] - medias['versicolor-sl']) ** 2).sum()/34
    variancias['versicolor-sw'] = ((dft[(dft['class'] == 'Iris-versicolor')]['sw'] - medias['versicolor-sw']) ** 2).sum()/34
    variancias['versicolor-pl'] = ((dft[(dft['class'] == 'Iris-versicolor')]['pl'] - medias['versicolor-pl']) ** 2).sum()/34
    variancias['versicolor-pw'] = ((dft[(dft['class'] == 'Iris-versicolor')]['pw'] - medias['versicolor-pw']) ** 2).sum()/34
    variancias['virginica-sl'] = ((dft[(dft['class'] == 'Iris-virginica')]['sl'] - medias['virginica-sl']) ** 2).sum()/34
    variancias['virginica-sw'] = ((dft[(dft['class'] == 'Iris-virginica')]['sw'] - medias['virginica-sw']) ** 2).sum()/34
    variancias['virginica-pl'] = ((dft[(dft['class'] == 'Iris-virginica')]['pl'] - medias['virginica-pl']) ** 2).sum()/34
    variancias['virginica-pw'] = ((dft[(dft['class'] == 'Iris-virginica')]['pw'] - medias['virginica-pw']) ** 2).sum()/34

# 1.3 Desvio padrão: A raiz da variáncia
def calcularDesvioPadrao(dft):
    desvioPadrao['setosa-sl'] = math.sqrt(variancias['setosa-sl'])
    desvioPadrao['setosa-sw'] = math.sqrt(variancias['setosa-sw'])
    desvioPadrao['setosa-pl'] = math.sqrt(variancias['setosa-pl'])
    desvioPadrao['setosa-pw'] = math.sqrt(variancias['setosa-pw'])
    desvioPadrao['versicolor-sl'] = math.sqrt(variancias['versicolor-sl'])
    desvioPadrao['versicolor-sw'] = math.sqrt(variancias['versicolor-sw'])
    desvioPadrao['versicolor-pl'] = math.sqrt(variancias['versicolor-pl'])
    desvioPadrao['versicolor-pw'] = math.sqrt(variancias['versicolor-pw'])
    desvioPadrao['virginica-sl'] = math.sqrt(variancias['virginica-sl'])
    desvioPadrao['virginica-sw'] = math.sqrt(variancias['virginica-sw'])
    desvioPadrao['virginica-pl'] = math.sqrt(variancias['virginica-pl'])
    desvioPadrao['virginica-pw'] = math.sqrt(variancias['virginica-pw'])

calcularMedias(dadosTreinamento)
calcularVariancias(dadosTreinamento)
calcularDesvioPadrao(dadosTreinamento)

def obterGaussiana(escala, media, desvioPadrao):
    return np.exp(-np.power(escala - media, 2.) / (2 * np.power(desvioPadrao, 2.)))

dfGaussianas = []
x = np.arange(0, 10, .001)
def montarGaussianas(caracteristica):
    for contador in np.arange(0, 11):
        dfGaussianas.insert(contador, obterGaussiana(contador, medias[caracteristica], desvioPadrao[caracteristica]))

montarGaussianas('setosa-sl')

gaussianaSetosaSl = norm(loc = medias['setosa-sl'], scale = desvioPadrao['setosa-sl'])
gaussianaVersicolorSl = norm(loc = medias['versicolor-sl'], scale = desvioPadrao['versicolor-sl'])
gaussianaVirginicaSl = norm(loc = medias['virginica-sl'], scale = desvioPadrao['virginica-sl'])

plt.plot(x, gaussianaSetosaSl.pdf(x), x, gaussianaVersicolorSl.pdf(x), x, gaussianaVirginicaSl.pdf(x))
plt.show()