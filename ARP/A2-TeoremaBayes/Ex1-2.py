# 1) Apresente o gráfico das funções de probabilidade: p(SL | Ci), p(SW | Ci), p(PL | Ci), p(SW |Ci)
# 2) Apresente o gráfico do modelo usando o classificador Bayesiano:
#       - p(Ci | SL, SW)
#       - p(Ci | SL, PL)
#       - p(Ci | SL, PW)
#       - p(Ci | SW, PL)
#       - p(Ci | SW, PW)
#       - p(Ci | PL, PW)
# Nota: as variáveis em questão são variáveis contínuas.

import math

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
dft = {}  # dataFrame de Treinamento
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
            medias[classCaract] = dft[(dft['class'] == iClass)][iCaracteristica['nome']].sum() / 35

def calcularVariancias():  # Variáncia: Somatorio de (Valor - media elevado ao quadrado) / número de elementos
    for iClass in classes:
        for iCaracteristica in caracteristicas:
            classCaract = iClass + '-' + iCaracteristica['nome']
            variancias[classCaract] = ((dft[(dft['class'] == iClass)][iCaracteristica['nome']] -
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
    global df, dft
    df = pd.read_csv('iris.data')
    dft = df.iloc[:35].append(df.iloc[50:85]).append(df.iloc[100:135]).copy()

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
ex1_2p()