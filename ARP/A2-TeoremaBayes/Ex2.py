# 1) Realize a classificação dos dados de teste usando o classificador do Problema 01.
# # Calcule e apresente a taxa de acerto de cada amostra de teste para cada classe.

import math
from array import array

import numpy as np
import pandas as pd
from matplotlib.mlab import bivariate_normal
from scipy.stats import norm
from mayavi import mlab
import matplotlib
from mpl_toolkits.mplot3d import axes3d
matplotlib.use("TkAgg", force=True)
import matplotlib.pyplot as plt

# Declaração de variáveis
df = {}  # dataFrame Original
dfTreinamento = {}  # dataFrame de treinamento
dfTeste = {}  # dataFrame de teste
medias = {}  # medias dos 35 primeiros de cada classe
variancias = {}  # variancias dos 35 primeiros de cada classe
desvioPadrao = {}  # desvio padrão dos 35 primeiros de cada classe

classes = {'setosa', 'versicolor', 'virginica'}
caracteristicas = [{'nome': 'sl', 'rangeInit': 2, 'rangeFinal': 10},
                   {'nome': 'sw', 'rangeInit': 0, 'rangeFinal': 6},
                   {'nome': 'pl', 'rangeInit': 0, 'rangeFinal': 9},
                   {'nome': 'pw', 'rangeInit': -1, 'rangeFinal': 4}]

grupoCaracteristicas = [{'char1': 'sl', 'char2': 'sw'},
                        {'char1': 'sl', 'char2': 'pl'},
                        {'char1': 'sl', 'char2': 'pw'},
                        {'char1': 'sw', 'char2': 'pl'},
                        {'char1': 'sw', 'char2': 'pw'},
                        {'char1': 'pl', 'char2': 'pw'}]

# 1) Achar gaussianas (distribuição normal) Exemplo: p(SL | Ci).
# 1.1 Achar media - 1.2 Achar variáncia - 1.3 Achar desvio padrão

def calcularMedias():
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

def calcularGaussiana(classCaract):
    return norm(loc=medias[classCaract], scale=desvioPadrao[classCaract])

def calcularGaussianaBinomial(x, y, classChar1, classChar2):
    return bivariate_normal(
        x, y, desvioPadrao[classChar1], desvioPadrao[classChar2], medias[classChar1], medias[classChar2])

def plotar():
    for iCaracteristica in caracteristicas:
        for iClass in classes:
            classCaract = iClass + '-' + iCaracteristica['nome']
            gaussiana = calcularGaussiana(classCaract)
            slRange = np.arange(iCaracteristica['rangeInit'], iCaracteristica['rangeFinal'], .001)
            plt.plot(slRange, gaussiana.pdf(slRange))
        plt.show()

def init():
    global df, dfTreinamento, dfTeste
    df = pd.read_csv('iris.data')
    dfTreinamento = df.iloc[:35].append(df.iloc[50:85]).append(df.iloc[100:135]).copy()
    dfTeste = df.iloc[35:50].append(df.iloc[85:100]).append(df.iloc[135:150]).copy()

def ex1_1p():
    init()
    calcularMedias(), calcularVariancias(), calcularDesvioPadrao(), plotar()

def ex1_2p():
    init()
    calcularMedias(), calcularVariancias(), calcularDesvioPadrao()

    for caracteristica in grupoCaracteristicas:
        char1 = caracteristica['char1']
        char2 = caracteristica['char2']
        biGaussiana = []
        i=0
        for iClass in classes:
            classChar1 = iClass + '-' + char1
            classChar2 = iClass + '-' + char2
            x, y = np.linspace(-2,10, 200), np.linspace(-2, 10, 200)
            X, Y = np.meshgrid(x, y)
            biGaussiana.insert(i,calcularGaussianaBinomial(X, Y, classChar1, classChar2))
            i = i+1

        mlab.figure(bgcolor=(1,1,1), fgcolor=(0.,0.,0.))

        mlab.surf(x, y, biGaussiana[0], colormap="Purples")
        mlab.surf(x, y, biGaussiana[1], colormap="Greys")
        mlab.surf(x, y, biGaussiana[2], colormap="Reds")
        mlab.axes(
            color=(1.0,1.0,1.0),
            nb_labels=4,
            xlabel=char1,
            ylabel=char2,
            x_axis_visibility=True,
            y_axis_visibility=True,
            z_axis_visibility=True,
            ranges=[-2, 10, -2, 10, 0, 1])
    mlab.show()

def ex2():
    init()
    calcularMedias(), calcularVariancias(), calcularDesvioPadrao()
    for iCaracteristica in caracteristicas:
        for iClass in classes:
            classCaract = iClass + '-' + iCaracteristica['nome']
            gaussiana = calcularGaussiana(classCaract)
            coluna = 'P({}|{})'.format(iCaracteristica['nome'], iClass)
            dfTeste[coluna] = gaussiana.pdf(dfTeste[iCaracteristica['nome']])

    dfTeste['Somas SL'] = (dfTeste['P(sl|versicolor)'] + dfTeste['P(sl|virginica)'] + dfTeste['P(sl|setosa)']) * 1/3
    dfTeste['Somas SW'] = (dfTeste['P(sw|versicolor)'] + dfTeste['P(sw|virginica)'] + dfTeste['P(sw|setosa)']) * 1/3
    dfTeste['Somas PL'] = (dfTeste['P(pl|versicolor)'] + dfTeste['P(pl|virginica)'] + dfTeste['P(pl|setosa)']) * 1/3
    dfTeste['Somas PW'] = (dfTeste['P(pw|versicolor)'] + dfTeste['P(pw|virginica)'] + dfTeste['P(pw|setosa)']) * 1/3

    dfTeste['Somas Gaussianas'] = (dfTeste['Somas SL'] * dfTeste['Somas SW'] * dfTeste['Somas PL'] * dfTeste['Somas PW'])

    dfTeste['P(Setosa| SL, SW, PL, PW)'] = ((dfTeste['P(sl|setosa)'] * dfTeste['P(sw|setosa)'] * dfTeste['P(pl|setosa)'] * dfTeste['P(pw|setosa)']) * 1/3) / dfTeste['Somas Gaussianas']
    dfTeste['P(Versicolor| SL, SW, PL, PW)'] = ((dfTeste['P(sl|versicolor)'] * dfTeste['P(sw|versicolor)'] * dfTeste['P(pl|versicolor)'] * dfTeste['P(pw|versicolor)']) * 1/3) /dfTeste['Somas Gaussianas']
    dfTeste['P(Virginica| SL, SW, PL, PW)'] = ((dfTeste['P(sl|virginica)'] * dfTeste['P(sw|virginica)'] * dfTeste['P(pl|virginica)'] * dfTeste['P(pw|virginica)']) * 1/3) /dfTeste['Somas Gaussianas']

    for index, row in dfTeste.iterrows():
        argmax = np.argmax([row['P(Setosa| SL, SW, PL, PW)'], row['P(Versicolor| SL, SW, PL, PW)'], row['P(Virginica| SL, SW, PL, PW)']])
        if(argmax == 0):
            argmax = 'setosa'
        elif(argmax == 1):
            argmax = 'versicolor'
        elif(argmax == 2):
            argmax = 'virginica'

        dfTeste.set_value(index, 'argMax', argmax)

    dfTeste['Accuracy'] = dfTeste['argMax'] == dfTeste['class']
    print(dfTeste)

ex2()