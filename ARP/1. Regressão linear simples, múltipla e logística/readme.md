# Regressão linear simples, múltipla e logística.

## Regressão linear simples


> A análide de regresão estuda o **relacionamento** entre uma variável
> **dependente** e outra **independente**. Este relacionamento é
> representado por um **modelo** matemático, isto é, por uma equação que
> relaciona. Tem como objetivo determinar a equação da reta ajustada.


__Problema de estimação de salário__

<table>
    <tr><td>X (Tempo experiência)</td><td>Y (Salário)</td></tr>
    <tr><td>2</td><td>2100</td></tr>
    <tr><td>5</td><td>4000</td></tr>
    <tr><td>8</td><td>6000</td></tr>
    <tr><td>10</td><td>14000</td></tr>
    <tr><td>6</td><td>8000</td></tr>
    <tr><td>7</td><td>5500</td></tr>
    <tr><td>1</td><td>2500</td></tr>
    <tr><td>5</td><td>7000</td></tr>
</table>

Um candidato com 8 anos de experiência pedindo salário de R$ 4000,00. Se
adequa ao salário da empresa ?

1. Descobrir qual é a relação entre a experiência e o salário ?

y = b0 + b1 x + e

y = variável que será explicada (dependente) e = Erro do modelo x =
Variável de entrada (independente)


E = Somatorio ~ = Média (- encima do y)

Para estimar: b¹ = Exy/Exx b0 = ~y-b¹~y

b¹ = 1013,9 b0 = 180,59

^Y = Quer dizer valor estimado (estimativa)

^Y = 180,59 + 1013,9 * 8 ^Y=8291,79

Capacidade de generalização: Discernir entre realidades diferentes.

Erro quadrático médio:

![EQM - Erro Quadrático Médio](images/EQM.svg?raw=true)


i = indice

    n
    E   (Yi - ^Yi)²

I = 0

  --------------------

            N

Yi = valor real ^Yi = valor estimado

> O modelo é bom ? Só utilizar o modelo de determinação R² R² = (Exy)² /
> ExxEyy = 0,9541
> * 0 = Não tem logica nenhum
> * 1 = Exato
> * \> 0.5 <0.7 é mais o menos
> * \> 0.7 é bom

<table>
    <tr><td>0</td><td>Não tem logica nenhum</td>
    <tr><td>1</td><td>0</td>
    <tr><td></td><td>1</td>
    <tr><td>8</td><td>2</td>
</table>


## Regressão linear múltipla

<table>
    <tr><td>X (Experiência)</td><td>Cursos</td><td>Y (Salário)</td></tr>
    <tr><td>2</td><td>0</td><td>2100</td></tr>
    <tr><td>5</td><td>1</td><td>4000</td></tr>
    <tr><td>8</td><td>2</td><td>6000</td></tr>
    <tr><td>10</td><td>5</td><td>14000</td></tr>
    <tr><td>6</td><td>1</td><td>8000</td></tr>
    <tr><td>7</td><td>3</td><td>5500</td></tr>
    <tr><td>1</td><td>1</td><td>2500</td></tr>
    <tr><td>5</td><td>4</td><td>7000</td></tr>
</table>

y = b0 + b1 x1 + b2x2

T= Transposta b = (xT x)-¹ xTy

b0 = 430,2 b1 = 364,3 b2 = 2210,6

## Logística

               -(b0+ b1x)

Y = 1 / 1 + e


teste estatistico regressão lso

