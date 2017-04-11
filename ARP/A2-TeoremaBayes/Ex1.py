# 1) Apresente o gráfico das funções de probabilidade: p(SL | Ci), p(SW | Ci), p(PL | Ci), p(SW |Ci)
# 2) E o gráfico do modelo de classificação p(Ci | SL, SW, PL, PW) usando o classificador Bayesiano.
# 3) Nota: as variáveis em questão são variáveis contínuas.

import pandas as pd
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

df = pd.read_csv('iris.data')


dadosTreinamento = df.iloc[:35, :].append(df.iloc[50:85, :]).append(df.iloc[100:135, :]).copy()

# 1) Gaussianas Ex. p(SL | Ci).
# Para achar as gaussianas (distribuição normal) teremos que achar a 1.1 media e o 1.3 desvio padrão
# Para achar o desvio padrão teremos que achar a 1.2 variáncia
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


# 1.2 Variáncia: Somatorio de (Valor - media elevado ao quadrado) / número de elementos
def calcularVariancias(dft):
    variancias['setosa-sl'] = ((dft[(dft['class'] == 'Iris-setosa')]['sl'] - medias['setosa-sl']) ** 2).sum() / 34
    variancias['setosa-sw'] = ((dft[(dft['class'] == 'Iris-setosa')]['sw'] - medias['setosa-sw']) ** 2).sum() / 34
    variancias['setosa-pl'] = ((dft[(dft['class'] == 'Iris-setosa')]['pl'] - medias['setosa-pl']) ** 2).sum() / 34
    variancias['setosa-pw'] = ((dft[(dft['class'] == 'Iris-setosa')]['pw'] - medias['setosa-pw']) ** 2).sum() / 34
    variancias['versicolor-sl'] = ((dft[(dft['class'] == 'Iris-versicolor')]['sl'] - medias[
        'versicolor-sl']) ** 2).sum() / 34
    variancias['versicolor-sw'] = ((dft[(dft['class'] == 'Iris-versicolor')]['sw'] - medias[
        'versicolor-sw']) ** 2).sum() / 34
    variancias['versicolor-pl'] = ((dft[(dft['class'] == 'Iris-versicolor')]['pl'] - medias[
        'versicolor-pl']) ** 2).sum() / 34
    variancias['versicolor-pw'] = ((dft[(dft['class'] == 'Iris-versicolor')]['pw'] - medias[
        'versicolor-pw']) ** 2).sum() / 34
    variancias['virginica-sl'] = ((dft[(dft['class'] == 'Iris-virginica')]['sl'] - medias[
        'virginica-sl']) ** 2).sum() / 34
    variancias['virginica-sw'] = ((dft[(dft['class'] == 'Iris-virginica')]['sw'] - medias[
        'virginica-sw']) ** 2).sum() / 34
    variancias['virginica-pl'] = ((dft[(dft['class'] == 'Iris-virginica')]['pl'] - medias[
        'virginica-pl']) ** 2).sum() / 34
    variancias['virginica-pw'] = ((dft[(dft['class'] == 'Iris-virginica')]['pw'] - medias[
        'virginica-pw']) ** 2).sum() / 34


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


gaussianaSetosaSl = norm(loc=medias['setosa-sl'], scale=desvioPadrao['setosa-sl'])
gaussianaVersicolorSl = norm(loc=medias['versicolor-sl'], scale=desvioPadrao['versicolor-sl'])
gaussianaVirginicaSl = norm(loc=medias['virginica-sl'], scale=desvioPadrao['virginica-sl'])

gaussianaSetosaSw = norm(loc=medias['setosa-sw'], scale=desvioPadrao['setosa-sw'])
gaussianaVersicolorSw = norm(loc=medias['versicolor-sw'], scale=desvioPadrao['versicolor-sw'])
gaussianaVirginicaSw = norm(loc=medias['virginica-sw'], scale=desvioPadrao['virginica-sw'])

gaussianaSetosaPl = norm(loc=medias['setosa-pl'], scale=desvioPadrao['setosa-pl'])
gaussianaVersicolorPl = norm(loc=medias['versicolor-pl'], scale=desvioPadrao['versicolor-pl'])
gaussianaVirginicaPl = norm(loc=medias['virginica-pl'], scale=desvioPadrao['virginica-pl'])

gaussianaSetosaPw = norm(loc=medias['setosa-pw'], scale=desvioPadrao['setosa-pw'])
gaussianaVersicolorPw = norm(loc=medias['versicolor-pw'], scale=desvioPadrao['versicolor-pw'])
gaussianaVirginicaPw = norm(loc=medias['virginica-pw'], scale=desvioPadrao['virginica-pw'])

slGraficos = plt
slRange = np.arange(0, 10, .001)
slGraficos.plot(slRange, gaussianaSetosaSl.pdf(slRange),
                slRange, gaussianaVersicolorSl.pdf(slRange),
                slRange, gaussianaVirginicaSl.pdf(slRange))
slGraficos.show()

swGraficos = plt
swRange = np.arange(0, 6, .001)
swGraficos.plot(swRange, gaussianaSetosaSw.pdf(swRange),
                swRange, gaussianaVersicolorSw.pdf(swRange),
                swRange, gaussianaVirginicaSw.pdf(swRange))
swGraficos.show()

plGraficos = plt
plRange = np.arange(0, 10, .001)
plGraficos.plot(plRange, gaussianaSetosaPl.pdf(plRange),
                plRange, gaussianaVersicolorPl.pdf(plRange),
                plRange, gaussianaVirginicaPl.pdf(plRange))
plGraficos.show()

pwGraficos = plt
pwRange = np.arange(-1, 4, .001)
pwGraficos.plot(pwRange, gaussianaSetosaPw.pdf(pwRange),
                pwRange, gaussianaVersicolorPw.pdf(pwRange),
                pwRange, gaussianaVirginicaPw.pdf(pwRange))
pwGraficos.show()
