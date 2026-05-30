import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração para exibir todas as colunas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Leitura dos arquivos CSV
df1 = pd.read_csv("c:/Users/Matheus Sarandy/Desktop/code/DesafioMathBigData/regiões atendidas.csv", sep=';')
df2 = pd.read_csv("c:/Users/Matheus Sarandy/Desktop/code/DesafioMathBigData/relaçãode redução de atrasos.csv", sep=';')
df3 = pd.read_csv("c:/Users/Matheus Sarandy/Desktop/code/DesafioMathBigData/tempo medio de entrega.csv", sep=';')

# Exibir os DataFrames formatados no terminal
print("\nVendas por região:")
print(df1.to_string(index=False))

print("\nRedução de atrasos:")
print(df2.to_string(index=False))

print("\nTempo médio de entrega:")
print(df3.to_string(index=False))

# =========================
# GRÁFICO 1
# Vendas por região
# =========================

dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]

plt.figure(figsize=(10, 5))

for i in range(len(df1)):
    plt.plot(
        dias,
        df1.loc[i, dias],
        marker='o',
        label=df1.loc[i, "Região"]
    )

plt.title("Vendas por Região Durante a Semana")
plt.xlabel("Dias")
plt.ylabel("Vendas")
plt.legend()
plt.grid(True)



# =========================
# GRÁFICO 2
# Investimento x redução
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    df2["Mês"],
    df2["Investimento (mil R$)"],
    marker='o',
    label="Investimento"
)

plt.plot(
    df2["Mês"],
    df2["Redução percentual de atrasos"],
    marker='o',
    label="Redução de atrasos"
)

plt.title("Investimento x Redução de Atrasos")
plt.xlabel("Mês")
plt.ylabel("Valores")
plt.legend()
plt.grid(True)



# =========================
# GRÁFICO 3
# Tempo por dia
# =========================

plt.figure(figsize=(10, 5))

plt.plot(
    df3["Dia"],
    df3["Tempo (min)"],
    marker='o'
)

plt.title("Tempo por Dia")
plt.xlabel("Dia")
plt.ylabel("Tempo (min)")
plt.grid(True)

plt.show()
plt.show()
plt.show()


# =====================================================
# FUNÇÃO PARA DESCOBRIR MELHOR AJUSTE
# =====================================================

def melhor_funcao(x, y):

    # Linear
    coef1 = np.polyfit(x, y, 1)
    func1 = np.poly1d(coef1)

    erro1 = np.mean((y - func1(x))**2)

    # Quadrática
    coef2 = np.polyfit(x, y, 2)
    func2 = np.poly1d(coef2)

    erro2 = np.mean((y - func2(x))**2)

    # Escolher menor erro
    if erro1 < erro2:
        return func1, "Função do 1º grau"
    else:
        return func2, "Função do 2º grau"

# =====================================================
# GRÁFICO 1 — VENDAS POR REGIÃO
# =====================================================

print("\n==============================")
print("PREVISÃO — VENDAS POR REGIÃO")
print("==============================")

dias = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta"]

x1 = np.array([1,2,3,4,5])

plt.figure(figsize=(12,6))

for i in range(len(df1)):

    y1 = df1.loc[i, dias].values
    regiao = df1.loc[i, "Região"]

    funcao, tipo = melhor_funcao(x1, y1)

    # Próximas previsões
    prox_x = np.array([6,7,8])

    previsoes = funcao(prox_x)

    print(f"\nRegião: {regiao}")
    print(f"Melhor ajuste: {tipo}")

    for j in range(3):
        print(f"Previsão {j+1}: {previsoes[j]:.2f}")

    # Gráfico
    x_total = np.concatenate((x1, prox_x))
    y_total = funcao(x_total)

    plt.plot(
        x_total,
        y_total,
        marker='o',
        label=f"{regiao} ({tipo})"
    )

plt.title("Previsão de Vendas por Região")
plt.xlabel("Dias")
plt.ylabel("Vendas")

plt.xticks(
    [1,2,3,4,5,6,7,8],
    ["Seg","Ter","Qua","Qui","Sex","P1","P2","P3"]
)

plt.legend()
plt.grid(True)

plt.show()

# =====================================================
# GRÁFICO 2 — REDUÇÃO DE ATRASOS
# =====================================================

print("\n==============================")
print("PREVISÃO — REDUÇÃO DE ATRASOS")
print("==============================")

x2 = np.array([1,2,3,4,5])

y2 = df2["Redução percentual de atrasos"].values

funcao2, tipo2 = melhor_funcao(x2, y2)

prox2 = np.array([6,7,8])

prev2 = funcao2(prox2)

print(f"\nMelhor ajuste: {tipo2}")

for i in range(3):
    print(f"Previsão {i+1}: {prev2[i]:.2f}%")

plt.figure(figsize=(12,6))

x_total2 = np.concatenate((x2, prox2))

y_total2 = funcao2(x_total2)

plt.plot(
    x_total2,
    y_total2,
    marker='o',
    label=tipo2
)

plt.scatter(x2, y2, label="Dados reais")

plt.title("Previsão de Redução de Atrasos")
plt.xlabel("Meses")
plt.ylabel("Redução (%)")

plt.xticks(
    [1,2,3,4,5,6,7,8],
    ["Jan","Fev","Mar","Abr","Mai","P1","P2","P3"]
)

plt.legend()
plt.grid(True)

plt.show()


# =====================================================
# DADOS DE REDUÇÃO DE ATRASOS
# =====================================================

x_red = np.array([1,2,3,4,5])

y_red = df2["Redução percentual de atrasos"].values

# Função matemática para previsão
coef = np.polyfit(x_red, y_red, 2)

funcao = np.poly1d(coef)

# Próximas previsões
prox = np.array([6,7,8,9])

previsoes = funcao(prox)

print("\nPREVISÕES DE REDUÇÃO")
print(f"P1: {previsoes[0]:.2f}%")
print(f"P2: {previsoes[1]:.2f}%")
print(f"P3: {previsoes[2]:.2f}%")
print(f"P4: {previsoes[3]:.2f}%")

# =====================================================
# DADOS DO TEMPO POR DIA
# =====================================================

dias = df3["Dia"]

tempos = df3["Tempo (min)"].values.astype(float)

# =====================================================
# APLICAR REDUÇÕES
# =====================================================

tempo_p1 = tempos * (1 - previsoes[0] / 100)

tempo_p2 = tempos * (1 - previsoes[1] / 100)

tempo_p3 = tempos * (1 - previsoes[2] / 100)

tempo_p4 = tempos * (1 - previsoes[3] / 100)

# =====================================================
# GRÁFICO
# =====================================================

plt.figure(figsize=(12,6))

# Linha original
plt.plot(
    dias,
    tempos,
    marker='o',
    linewidth=3,
    label="Original"
)

# P1
plt.plot(
    dias,
    tempo_p1,
    marker='o',
    label=f"P1 ({previsoes[0]:.1f}%)"
)

# P2
plt.plot(
    dias,
    tempo_p2,
    marker='o',
    label=f"P2 ({previsoes[1]:.1f}%)"
)

# P3
plt.plot(
    dias,
    tempo_p3,
    marker='o',
    label=f"P3 ({previsoes[2]:.1f}%)"
)

# P4
plt.plot(
    dias,
    tempo_p4,
    marker='o',
    label=f"P4 ({previsoes[3]:.1f}%)"
)

# =====================================================
# CONFIGURAÇÕES
# =====================================================

plt.title("Previsão de Tempo com Redução de Atrasos")

plt.xlabel("Dias")

plt.ylabel("Tempo (min)")

plt.grid(True)

plt.legend()

plt.show()

print("\nPREVISÃO DE TEMPOS COM REDUÇÃO DE ATRASOS")
# =========================
# MÉDIA DO TEMPO
# =========================

media_tempo = df3["Tempo (min)"].mean()

print(f"Média semanal atual: {media_tempo:.2f} minutos")

# =========================
# META DA EMPRESA
# =========================

meta = 48

# =========================
# CÁLCULO DO PERCENTUAL
# =========================

if media_tempo > meta:

    diferenca = media_tempo - meta

    percentual_reducao = (diferenca / media_tempo) * 100

    print(f"\nA empresa considera aceitável até {meta} minutos.")

    print(
        f"É necessário reduzir "
        f"{percentual_reducao:.2f}% "
        f"do tempo médio para alcançar a meta."
    )

else:

    print("\nA média já está dentro da meta da empresa.")