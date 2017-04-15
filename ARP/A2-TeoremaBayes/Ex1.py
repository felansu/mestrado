# 1) Apresente o gráfico das funções de probabilidade: p(SL | Ci), p(SW | Ci), p(PL | Ci), p(SW |Ci)
# 2) Apresente o gráfico do modelo de classificação p(Ci | SL, SW, PL, PW) usando o classificador Bayesiano.
# 3) Nota: as variáveis em questão são variáveis contínuas.

import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = {}  # dataFrame Original
dfTreinamento = {}  # dataFrameTreinamento

# 1) Achar gaussianas (distribuição normal) Ex. p(SL | Ci).
# 1.1 Achar media - 1.2 Achar variáncia - 1.3 Achar desvio padrão

classes = {'setosa', 'versicolor', 'virginica'}
caracteristicas = [{'nome': 'sl', 'rangeInit': 2, 'rangeFinal': 10}, {'nome': 'sw', 'rangeInit': 0, 'rangeFinal': 6},
                   {'nome': 'pl', 'rangeInit': 0, 'rangeFinal': 9}, {'nome': 'pw', 'rangeInit': -1, 'rangeFinal': 4}]
medias = {}
variancias = {}
desvioPadrao = {}

def calcularMedias():
    print('oi')
    for iClass in classes:
        for iCaracteristica in caracteristicas:
            classCaract = iClass + '-' + iCaracteristica['nome']
            medias[classCaract] = dfTreinamento[(dfTreinamento['class'] == iClass)][iCaracteristica['nome']].sum() / 35

def calcularVariancias():  # Variáncia: Somatorio de (Valor - media elevado ao quadrado) / número de elementos
    for iClass in classes:
        for iCaracteristica in caracteristicas:
            classCaract = iClass + '-' + iCaracteristica['nome']
            variancias[classCaract] = ((dfTreinamento[(dfTreinamento['class'] == iClass)][iCaracteristica['nome']] -
                                        medias[
                                            classCaract]) ** 2).sum() / 34

def calcularDesvioPadrao():  # Desvio padrão: A raiz da variáncia
    for iClass in classes:
        for iCaracteristica in caracteristicas:
            classCaract = iClass + '-' + iCaracteristica['nome']
            desvioPadrao[classCaract] = math.sqrt(variancias[classCaract])

def plotar():
    for iCaracteristica in caracteristicas:
        for iClass in classes:
            classCaract = iClass + '-' + iCaracteristica['nome']
            gaussiana = norm(loc=medias[classCaract], scale=desvioPadrao[classCaract])
            slRange = np.arange(iCaracteristica['rangeInit'], iCaracteristica['rangeFinal'], .001)
            plt.plot(slRange, gaussiana.pdf(slRange))
        plt.show()

def main():
    global df, dfTreinamento
    df = pd.read_csv('iris.data')
    dft = df.iloc[:35, :].append(df.iloc[50:85, :]).append(df.iloc[100:135, :]).copy()
    calcularMedias(), calcularVariancias(), calcularDesvioPadrao(), plotar()

main()
