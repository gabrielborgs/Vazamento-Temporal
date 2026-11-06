"""Gera a imagem do post a partir de resultados.csv (rode vazamento_temporal.py antes)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle

r = pd.read_csv("resultados.csv")
r["infl"] = (r["aleatoria"] - r["tempo_com_folga"]) * 100            # em pontos de AUC (×100)
g = r.groupby("deriva")["infl"].agg(["mean", "min", "max", "count"])
rot = {0.0: "Rede estável", 0.01: "Piora moderada", 0.03: "Piora forte"}

SURF, INK, INK2, MUTED, AZUL = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#2a78d6"
W, H = 1200, 627
fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor=SURF)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
def t(x, y, s, size, color=INK, weight="normal", ha="left"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va="center", family="DejaVu Sans")

t(70, 70, "A divisão aleatória infla o resultado", 30, INK, "bold")
t(70, 114, "Quanto o AUC no teste fica maior do que numa divisão por tempo (mesmo modelo, mesmos atributos)", 14.5, INK2)
t(1130, 34, "Dados sintéticos", 11, MUTED, ha="right")

base_y, topo = 470, 190                     # linha de base dos valores (zero) e altura reservada
escala = (base_y - topo) / 3.0              # 3 pontos de AUC = altura toda
larg, gap = 220, 110
x_ini = 70 + (1060 - (3 * larg + 2 * gap)) / 2
ax.add_patch(Rectangle((70, base_y), 1060, 1.5, fc="#d9d8d3", ec="none"))
for i, (d, linha) in enumerate(g.iterrows()):
    x = x_ini + i * (larg + gap)
    h = linha["mean"] * escala
    ax.add_patch(FancyBboxPatch((x, base_y - h), larg, h, boxstyle="round,pad=0,rounding_size=6", fc=AZUL, ec="none"))
    ax.add_patch(Rectangle((x, base_y - 8), larg, 8, fc=AZUL, ec="none"))       # base reta, ponta de cima arredondada
    # faixa mín–máx das repetições
    xc = x + larg / 2
    ax.plot([xc, xc], [base_y - linha["max"] * escala, base_y - linha["min"] * escala], color=INK2, lw=1.6, solid_capstyle="butt")
    for v in (linha["max"], linha["min"]):
        ax.plot([xc - 9, xc + 9], [base_y - v * escala] * 2, color=INK2, lw=1.6)
    t(xc, base_y - linha["max"] * escala - 26, f"+{linha['mean']:.1f} pt".replace(".", ","), 22, INK, "bold", ha="center")
    t(xc, base_y + 30, rot[d], 16, INK, "bold", ha="center")
    t(xc, base_y + 56, f"{int(linha['count'])} repetições", 12.5, INK2, ha="center")

t(70, 585, "Barra = média; traço = menor e maior valor entre as repetições. pt = pontos de AUC (×100).", 12.5, INK2)
fig.savefig("grafico_vazamento.png", dpi=100, facecolor=SURF)
print(g.round(2))
