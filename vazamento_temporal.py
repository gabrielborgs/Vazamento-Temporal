"""
Vazamento temporal: quanto a forma de dividir treino e teste infla o resultado.

Cenário (dados 100% sintéticos): um painel de "células" com um indicador de qualidade diário.
Pergunta: "a média dos próximos 7 dias vai ficar abaixo da meta?"  →  o alvo de cada linha olha 7 dias para a frente.

Três formas de dividir os dados, com o MESMO modelo e os MESMOS atributos:
  1. aleatória                 : sorteia linhas para treino e teste (o que muita gente faz por hábito)
  2. por tempo, sem folga      : treina no passado, testa no futuro
  3. por tempo, com folga      : igual à 2, mas descarta do treino os últimos 7 dias antes do corte,
                                 para que nenhum alvo de treino enxergue o período de teste

Uso:  python vazamento_temporal.py            (imprime a tabela e grava resultados.csv)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

N_CELULAS, N_DIAS = 400, 240
JANELA, HORIZONTE = 14, 7          # atributos olham 14 dias para trás; o alvo, 7 para a frente
META = 94.0
FRACAO_TESTE = 0.25
SEEDS = range(10)


def gerar_painel(seed: int, deriva: float = 0.010) -> pd.DataFrame:
    """Indicador diário por célula: nível próprio + ruído autocorrelacionado + episódios de degradação + piora gradual da rede."""
    rng = np.random.default_rng(seed)
    nivel = rng.normal(96.5, 1.6, N_CELULAS)
    q = np.zeros((N_CELULAS, N_DIAS))
    ruido = np.zeros(N_CELULAS)
    em_episodio = np.zeros(N_CELULAS, dtype=int)
    for t in range(N_DIAS):
        ruido = 0.85 * ruido + rng.normal(0, 0.9, N_CELULAS)
        prob_episodio = 0.006 + deriva * (t / N_DIAS)       # a rede piora com o tempo: o futuro não se parece com o passado
        novo = (em_episodio == 0) & (rng.random(N_CELULAS) < prob_episodio)
        em_episodio[novo] = rng.integers(4, 18, novo.sum())
        queda = np.where(em_episodio > 0, rng.normal(7, 2, N_CELULAS), 0.0)
        em_episodio = np.maximum(em_episodio - 1, 0)
        q[:, t] = np.clip(nivel + ruido - queda, 0, 100)
    return pd.DataFrame(q)


def atributos_e_alvo(q: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por célula × dia: atributos da janela de 14 dias e alvo = média dos 7 dias seguintes abaixo da meta."""
    v = q.to_numpy()
    linhas = []
    for t in range(JANELA, N_DIAS - HORIZONTE):
        w = v[:, t - JANELA + 1: t + 1]
        f = v[:, t + 1: t + 1 + HORIZONTE]
        linhas.append(pd.DataFrame({
            "dia": t, "celula": np.arange(len(v)),
            "media": w.mean(1), "minimo": w.min(1), "desvio": w.std(1), "ultimo": w[:, -1],
            "tendencia": w[:, -7:].mean(1) - w[:, :7].mean(1), "dias_ruins": (w[:, -7:] < META).sum(1),
            "alvo": (f.mean(1) < META).astype(int),
        }))
    return pd.concat(linhas, ignore_index=True)


ATRIBUTOS = ["media", "minimo", "desvio", "ultimo", "tendencia", "dias_ruins"]


def avaliar(d: pd.DataFrame, treino: pd.Index, teste: pd.Index, seed: int) -> float:
    m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=seed)
    m.fit(d.loc[treino, ATRIBUTOS], d.loc[treino, "alvo"])
    return float(roc_auc_score(d.loc[teste, "alvo"], m.predict_proba(d.loc[teste, ATRIBUTOS])[:, 1]))


def rodar(seed: int, deriva: float = 0.010) -> dict:
    d = atributos_e_alvo(gerar_painel(seed, deriva))
    corte = int(d["dia"].min() + (1 - FRACAO_TESTE) * (d["dia"].max() - d["dia"].min()))
    teste_t = d.index[d["dia"] > corte]
    # 1. aleatória: mesmo tamanho de teste, linhas sorteadas
    rng = np.random.default_rng(seed)
    sorteio = rng.permutation(d.index.to_numpy())
    n_teste = len(teste_t)
    r = avaliar(d, pd.Index(sorteio[n_teste:]), pd.Index(sorteio[:n_teste]), seed)
    # 2. por tempo, sem folga
    t_sem = avaliar(d, d.index[d["dia"] <= corte], teste_t, seed)
    # 3. por tempo, com folga do tamanho do horizonte
    t_com = avaliar(d, d.index[d["dia"] <= corte - HORIZONTE], teste_t, seed)
    return {"seed": seed, "deriva": deriva, "aleatoria": r, "tempo_sem_folga": t_sem, "tempo_com_folga": t_com, "taxa_positivos": float(d["alvo"].mean())}


DERIVAS = [0.0, 0.010, 0.030]      # 0 = rede estável; quanto maior, mais o futuro difere do passado
SEEDS_SENSIBILIDADE = range(6)


if __name__ == "__main__":
    res = pd.DataFrame([rodar(s, d) for d in DERIVAS for s in (SEEDS if d == 0.010 else SEEDS_SENSIBILIDADE)])
    res.to_csv("resultados.csv", index=False)
    cols = ["aleatoria", "tempo_sem_folga", "tempo_com_folga"]
    base = res[res["deriva"] == 0.010]
    print(f"AUC no teste, cenário base ({len(base)} repetições; taxa média de positivos {base['taxa_positivos'].mean():.1%})")
    print(base[cols].agg(["mean", "std"]).T.round(3).to_string())
    print("\nSensibilidade: o quanto a divisão aleatória infla o AUC (aleatória − por tempo com folga), por deriva da rede")
    res["inflacao"] = res["aleatoria"] - res["tempo_com_folga"]
    print(res.groupby("deriva")[cols + ["inflacao"]].mean().round(3).to_string())
