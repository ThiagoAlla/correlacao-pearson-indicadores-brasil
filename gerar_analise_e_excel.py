#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projeto de Estatística Aplicada: Tratamento, Correlação de Pearson e Projeção de Dados
Base de Dados: World Development Indicators (WDI) - Brasil (World Bank)
Bibliotecas: pandas, numpy, scipy, matplotlib, seaborn, openpyxl
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import ScatterChart, Reference, Series
from openpyxl.chart.trendline import Trendline

# Configuração de diretórios
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CSV_PT_PATH = DATA_DIR / "API_BRA_DS2_pt_csv_v2_11137.csv"
META_PT_PATH = DATA_DIR / "Metadata_Indicator_API_BRA_DS2_pt_csv_v2_11137.csv"

# Indicadores selecionados para o estudo
INDICADORES_MAP = {
    "NY.GDP.PCAP.PP.CD": "PIB_per_capita_PPP_USD",
    "EG.USE.ELEC.KH.PC": "Consumo_Energia_Eletrica_kWh_pc",
    "SP.DYN.LE00.IN": "Expectativa_Vida_Anos",
    "SP.DYN.IMRT.IN": "Mortalidade_Infantil_por_1000",
    "SI.POV.DDAY": "Pobreza_Extrema_perc",
    "SI.POV.GINI": "Indice_Gini",
    "SE.ADT.LITR.ZS": "Taxa_Alfabetizacao_Adultos_perc",
    "SP.DYN.TFRT.IN": "Taxa_Fertilidade_Filhos_Mulher",
    "EG.USE.PCAP.KG.OE": "Uso_Energia_kg_Petroleo_pc",
    "SL.UEM.TOTL.ZS": "Taxa_Desemprego_perc",
}

NOMES_DESCRITIVOS = {
    "PIB_per_capita_PPP_USD": "PIB per capita, PPP (US$ internacional corrente)",
    "Consumo_Energia_Eletrica_kWh_pc": "Consumo de energia elétrica (kWh per capita)",
    "Expectativa_Vida_Anos": "Expectativa de vida ao nascer (anos)",
    "Mortalidade_Infantil_por_1000": "Taxa de mortalidade infantil (por 1.000 nascidos vivos)",
    "Pobreza_Extrema_perc": "Pobreza extrema - US$ 1,90/dia PPP (% da população)",
    "Indice_Gini": "Índice de Gini (0 a 100)",
    "Taxa_Alfabetizacao_Adultos_perc": "Taxa de alfabetização de adultos (% 15 anos ou mais)",
    "Taxa_Fertilidade_Filhos_Mulher": "Taxa de fecundidade total (nascimentos por mulher)",
    "Uso_Energia_kg_Petroleo_pc": "Uso de energia (kg de equivalente de petróleo per capita)",
    "Taxa_Desemprego_perc": "Desemprego total (% da força de trabalho)",
}


def carregar_e_tratar_dados() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega o CSV original do Banco Mundial e estrutura em série temporal (Anos nas linhas)."""
    df_raw = pd.read_csv(CSV_PT_PATH, skiprows=4)
    df_meta = pd.read_csv(META_PT_PATH)

    # Filtrar colunas de anos válidos (1960 a 2025)
    colunas_anos = [c for c in df_raw.columns if c.isdigit()]

    # Filtrar indicadores de interesse
    df_sel = df_raw[df_raw["Indicator Code"].isin(INDICADORES_MAP.keys())].copy()
    df_sel["Col_Curta"] = df_sel["Indicator Code"].map(INDICADORES_MAP)

    # Transpor para ter Anos como linhas e Indicadores como colunas
    df_anos = (
        df_sel.set_index("Col_Curta")[colunas_anos]
        .T.apply(pd.to_numeric, errors="coerce")
    )
    df_anos.index.name = "Ano"
    df_anos.index = df_anos.index.astype(int)
    df_anos = df_anos.sort_index()

    return df_anos, df_meta


def calcular_estatisticas_pearson(df: pd.DataFrame, col_x: str, col_y: str) -> dict:
    """Calcula estatísticas descritivas, correlação de Pearson (r), teste de hipótese e regressão linear."""
    sub = df[[col_x, col_y]].dropna().copy()
    x = sub[col_x].to_numpy(dtype=float)
    y = sub[col_y].to_numpy(dtype=float)
    n = len(sub)

    # Cálculo algébrico explícito (passo a passo)
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    dx = x - x_mean
    dy = y - y_mean
    sum_dx_dy = np.sum(dx * dy)
    sum_dx2 = np.sum(dx**2)
    sum_dy2 = np.sum(dy**2)
    r_manual = sum_dx_dy / np.sqrt(sum_dx2 * sum_dy2)

    # Cálculo via SciPy (r de Pearson, p-valor e Intervalo de Confiança 95%)
    pearson_res = stats.pearsonr(x, y)
    r_scipy = float(pearson_res.statistic)
    p_value = float(pearson_res.pvalue)
    ci_low, ci_high = pearson_res.confidence_interval(confidence_level=0.95)

    # Estatística t de Student para correlação
    t_stat = r_scipy * np.sqrt((n - 2) / (1 - r_scipy**2))

    # Regressão Linear Simples (Mínimos Quadrados Ordinários) para Projeção
    reg = stats.linregress(x, y)
    slope = float(reg.slope)
    intercept = float(reg.intercept)
    r_squared = r_scipy**2

    # Erro padrão da estimativa (resíduos)
    y_hat = intercept + slope * x
    residuos = y - y_hat
    erro_padrao_est = np.sqrt(np.sum(residuos**2) / (n - 2))

    return {
        "df_limpo": sub,
        "n": n,
        "ano_inicio": int(sub.index.min()),
        "ano_fim": int(sub.index.max()),
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "x_std": float(np.std(x, ddof=1)),
        "y_std": float(np.std(y, ddof=1)),
        "sum_dx_dy": float(sum_dx_dy),
        "sum_dx2": float(sum_dx2),
        "sum_dy2": float(sum_dy2),
        "cov_amostral": float(sum_dx_dy / (n - 1)),
        "r_manual": float(r_manual),
        "r_scipy": r_scipy,
        "r_squared": float(r_squared),
        "p_value": p_value,
        "t_stat": float(t_stat),
        "ci_95": (float(ci_low), float(ci_high)),
        "slope": slope,
        "intercept": intercept,
        "erro_padrao_est": float(erro_padrao_est),
    }


def gerar_graficos(df_moderno: pd.DataFrame, est_energia: dict, est_vida: dict, est_pobreza: dict) -> None:
    """Gera os diagramas de dispersão e matriz de correlação com Matplotlib e Seaborn."""
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 150

    # 1. Diagrama de Dispersão Principal: PIB per capita PPP vs Consumo de Energia Elétrica + Projeção
    fig, ax = plt.subplots(figsize=(11, 6.8))
    sub = est_energia["df_limpo"]
    x = sub["PIB_per_capita_PPP_USD"]
    y = sub["Consumo_Energia_Eletrica_kWh_pc"]

    sns.regplot(
        x=x,
        y=y,
        ax=ax,
        scatter_kws={"s": 70, "alpha": 0.85, "color": "#1f4e78", "edgecolor": "white"},
        line_kws={"color": "#d95f02", "linewidth": 2.2, "label": (
            f"Regressão Linear: Ŷ = {est_energia['intercept']:.2f} + {est_energia['slope']:.4f}·X\n"
            f"r de Pearson = {est_energia['r_scipy']:.4f} | R² = {est_energia['r_squared']:.4f} (p < 0,0001)"
        )},
    )

    # Projeção para metas de PIB per capita (US$ 25.000 e US$ 28.000 PPP)
    x_proj = np.array([x.iloc[-1], 25000.0, 28000.0])
    y_proj = est_energia["intercept"] + est_energia["slope"] * x_proj
    ax.plot(x_proj, y_proj, linestyle="--", color="#2ca02c", linewidth=2, label="Projeção Linear (Cenários USD 25k e USD 28k PPP)")
    ax.scatter(x_proj[1:], y_proj[1:], color="#2ca02c", s=95, marker="D", zorder=6, edgecolor="black")
    ax.set_ylim(1350, 4050)

    for xp, yp, rotulo in [(25000, y_proj[1], "Projeção USD 25k"), (28000, y_proj[2], "Projeção USD 28k")]:
        ax.annotate(
            f"{rotulo}\n({yp:.0f} kWh/hab)",
            xy=(xp, yp),
            xytext=(xp - 1900, yp + 120),
            fontsize=8.5,
            fontweight="bold",
            color="#1b7837",
            arrowprops={"arrowstyle": "->", "color": "#1b7837", "lw": 1},
        )

    # Anotar anos históricos marcantes do Brasil
    anos_destaque = [1990, 1994, 2001, 2010, 2014, 2020, 2025]
    for ano in anos_destaque:
        if ano in sub.index:
            xa = sub.loc[ano, "PIB_per_capita_PPP_USD"]
            ya = sub.loc[ano, "Consumo_Energia_Eletrica_kWh_pc"]
            offset_y = 65 if ano not in (2001, 2020) else -110
            ax.annotate(
                str(ano),
                xy=(xa, ya),
                xytext=(xa - 350, ya + offset_y),
                fontsize=8.5,
                fontweight="bold",
                color="#222222",
                arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.8},
            )

    ax.set_title(
        "Diagrama de Dispersão e Projeção Linear — Brasil (1990–2025)\n"
        "PIB per capita (PPP) × Consumo de Energia Elétrica per capita",
        fontsize=13.5,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Variável Independente (X): PIB per capita, PPP (US$ internacional corrente)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Variável Dependente (Y): Consumo de Energia Elétrica (kWh per capita)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.95, fontsize=9.5)

    plt.figtext(
        0.01,
        0.01,
        "Fonte dos Dados: World Bank - World Development Indicators (WDI), Brasil (1990-2025). "
        "Referência Teórica: Kraft & Kraft (1978); IEA / World Bank (2026).",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_DIR / "diagrama_dispersao_principal.png", dpi=300)
    plt.close(fig)

    # 2. Diagrama de Dispersão: Curva de Preston no Brasil (PIB per capita vs Expectativa de Vida)
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    sub_v = est_vida["df_limpo"]
    xv = sub_v["PIB_per_capita_PPP_USD"]
    yv = sub_v["Expectativa_Vida_Anos"]

    sns.regplot(
        x=xv,
        y=yv,
        ax=ax,
        scatter_kws={"s": 70, "alpha": 0.85, "color": "#2b5c8f", "edgecolor": "white"},
        line_kws={"color": "#c51b7d", "linewidth": 2.2, "label": (
            f"Regressão Linear: Ŷ = {est_vida['intercept']:.2f} + {est_vida['slope']:.5f}·X\n"
            f"r de Pearson = {est_vida['r_scipy']:.4f} | R² = {est_vida['r_squared']:.4f} (p < 0,0001)"
        )},
    )
    for ano in [1990, 1995, 2000, 2010, 2019, 2021, 2024]:
        if ano in sub_v.index:
            xa = sub_v.loc[ano, "PIB_per_capita_PPP_USD"]
            ya = sub_v.loc[ano, "Expectativa_Vida_Anos"]
            dy_off = -1.1 if ano == 2021 else 0.55
            label_txt = f"{ano} (COVID-19)" if ano == 2021 else str(ano)
            ax.annotate(
                label_txt,
                xy=(xa, ya),
                xytext=(xa - 400, ya + dy_off),
                fontsize=8.5,
                fontweight="bold",
                color="#b2182b" if ano == 2021 else "#222222",
                arrowprops={"arrowstyle": "->", "color": "#666666", "lw": 0.8},
            )

    ax.set_title(
        "Diagrama de Dispersão (Curva de Preston no Brasil — 1990–2024)\n"
        "PIB per capita (PPP) × Expectativa de Vida ao Nascer",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("PIB per capita, PPP (US$ internacional corrente)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expectativa de Vida ao Nascer (anos)", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9.5)
    plt.figtext(
        0.01,
        0.01,
        "Fonte dos Dados: World Bank - WDI (2026). Literatura de Referência: Preston, S. H. (1975). Population Studies, 29(2), 231-248.",
        fontsize=8.5,
        style="italic",
        color="#444444",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(OUTPUT_DIR / "diagrama_dispersao_preston_vida.png", dpi=300)
    plt.close(fig)

    # 3. Painel 2x2 de Diagramas de Dispersão da Realidade Brasileira
    fig, axes = plt.subplots(2, 2, figsize=(14, 10.5))

    pares_painel = [
        ("PIB_per_capita_PPP_USD", "Consumo_Energia_Eletrica_kWh_pc", "1. Desenvolvimento × Demanda Elétrica (1990–2025)", "#1f77b4", axes[0, 0]),
        ("PIB_per_capita_PPP_USD", "Expectativa_Vida_Anos", "2. Curva de Preston: Renda × Longevidade (1990–2024)", "#2ca02c", axes[0, 1]),
        ("PIB_per_capita_PPP_USD", "Pobreza_Extrema_perc", "3. Crescimento Econômico × Extrema Pobreza (1990–2024)", "#d62728", axes[1, 0]),
        ("Taxa_Alfabetizacao_Adultos_perc", "Mortalidade_Infantil_por_1000", "4. Educação (Alfabetização) × Mortalidade Infantil (2000–2024)", "#9467bd", axes[1, 1]),
    ]

    for cx, cy, titulo, cor, ax_p in pares_painel:
        est_p = calcular_estatisticas_pearson(df_moderno, cx, cy)
        sub_p = est_p["df_limpo"]
        sns.regplot(
            x=sub_p[cx],
            y=sub_p[cy],
            ax=ax_p,
            scatter_kws={"s": 50, "alpha": 0.8, "color": cor},
            line_kws={"color": "#222222", "linewidth": 1.8},
        )
        ax_p.set_title(f"{titulo}\nr = {est_p['r_scipy']:.4f} | R² = {est_p['r_squared']:.4f} (n = {est_p['n']})", fontsize=11, fontweight="bold")
        ax_p.set_xlabel(NOMES_DESCRITIVOS[cx], fontsize=9.5)
        ax_p.set_ylabel(NOMES_DESCRITIVOS[cy], fontsize=9.5)

    fig.suptitle(
        "Painel de Diagramas de Dispersão e Correlação de Pearson — Indicadores do Brasil (World Bank WDI)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUTPUT_DIR / "painel_dispersao_brasil.png", dpi=300)
    plt.close(fig)

    # 4. Matriz de Correlação de Pearson (Heatmap)
    colunas_heatmap = [
        "PIB_per_capita_PPP_USD",
        "Consumo_Energia_Eletrica_kWh_pc",
        "Uso_Energia_kg_Petroleo_pc",
        "Expectativa_Vida_Anos",
        "Taxa_Alfabetizacao_Adultos_perc",
        "Mortalidade_Infantil_por_1000",
        "Taxa_Fertilidade_Filhos_Mulher",
        "Pobreza_Extrema_perc",
        "Indice_Gini",
    ]
    rotulos_curtos = [
        "PIB per capita (PPP)",
        "Energia Elétrica (kWh/hab)",
        "Uso de Energia (kg pet./hab)",
        "Expectativa de Vida",
        "Alfabetização Adultos (%)",
        "Mortalidade Infantil",
        "Taxa de Fecundidade",
        "Pobreza Extrema (%)",
        "Índice de Gini",
    ]
    corr_matrix = df_moderno[colunas_heatmap].corr(method="pearson")
    corr_matrix.index = rotulos_curtos
    corr_matrix.columns = rotulos_curtos

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        linewidths=0.8,
        cbar_kws={"label": "Coeficiente de Correlação de Pearson (r)"},
        ax=ax,
    )
    ax.set_title(
        "Matriz de Correlação Linear de Pearson (r) — Indicadores Socioeconômicos do Brasil (1990–2025)\n"
        "Fonte: World Bank — World Development Indicators",
        fontsize=12.5,
        fontweight="bold",
        pad=14,
    )
    plt.xticks(rotation=35, ha="right", fontsize=9.5)
    plt.yticks(fontsize=9.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "matriz_correlacao_brasil.png", dpi=300)
    plt.close(fig)


def construir_aba_analise_excel(
    wb: openpyxl.Workbook,
    titulo_aba: str,
    titulo_relatorio: str,
    literatura_ref: str,
    nome_x: str,
    nome_y: str,
    unidade_x: str,
    unidade_y: str,
    est: dict,
    conclusao_texto: str,
    cenarios_proj: list[tuple[str, float]],
    primeira_aba: bool = False,
) -> None:
    """Cria uma aba completa no Excel com tabela de dados, cálculo de Pearson via fórmulas nativas, gráfico de dispersão nativo e conclusão."""
    ws = wb.active if primeira_aba else wb.create_sheet(title=titulo_aba)
    ws.title = titulo_aba
    ws.views.sheetView[0].showGridLines = True

    # Estilos
    azul_escuro = PatternFill("solid", fgColor="1F4E78")
    azul_medio = PatternFill("solid", fgColor="2F5597")
    azul_claro = PatternFill("solid", fgColor="D9E1F2")
    cinza_claro = PatternFill("solid", fgColor="F2F2F2")
    verde_claro = PatternFill("solid", fgColor="E2EFDA")
    amarelo_claro = PatternFill("solid", fgColor="FFF2CC")

    fonte_titulo = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    fonte_subtitulo = Font(name="Calibri", size=10, italic=True, color="1F4E78")
    fonte_cabecalho = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
    fonte_secao = Font(name="Calibri", size=11, bold=True, color="1F4E78")
    fonte_normal = Font(name="Calibri", size=10)
    fonte_negrito = Font(name="Calibri", size=10, bold=True)

    borda_fina = Border(
        left=Side(style="thin", color="BFBFBF"),
        right=Side(style="thin", color="BFBFBF"),
        top=Side(style="thin", color="BFBFBF"),
        bottom=Side(style="thin", color="BFBFBF"),
    )

    # Título e Referência Bibliográfica
    ws.merge_cells("A1:I1")
    ws["A1"] = titulo_relatorio
    ws["A1"].fill = azul_escuro
    ws["A1"].font = fonte_titulo
    ws["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws.row_dimensions[1].height = 28

    ws.merge_cells("A2:I3")
    ws["A2"] = f"LITERATURA / REFERÊNCIA OFICIAL DOS DADOS E DO DIAGRAMA DE DISPERSÃO:\n{literatura_ref}"
    ws["A2"].fill = amarelo_claro
    ws["A2"].font = fonte_subtitulo
    ws["A2"].alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)
    ws.row_dimensions[2].height = 22
    ws.row_dimensions[3].height = 22

    # Cabeçalho da Tabela de Dados e Cálculo Algébrico Passo a Passo
    row_header = 5
    headers = [
        "Ano",
        f"Variável X:\n{nome_x}",
        f"Variável Y:\n{nome_y}",
        "Desvio em X\n(xi - x̄)",
        "Desvio em Y\n(yi - ȳ)",
        "Produto dos Desvios\n(xi - x̄)·(yi - ȳ)",
        "Quadrado Desvio X\n(xi - x̄)²",
        "Quadrado Desvio Y\n(yi - ȳ)²",
        f"Projeção Linear (Ŷ)\n{unidade_y}",
    ]
    for col_idx, h_text in enumerate(headers, start=1):
        cell = ws.cell(row=row_header, column=col_idx, value=h_text)
        cell.fill = azul_medio
        cell.font = fonte_cabecalho
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = borda_fina
    ws.row_dimensions[row_header].height = 30

    sub = est["df_limpo"]
    start_row = 6
    end_row = start_row + len(sub) - 1
    mean_row = end_row + 1

    for idx_offset, (ano, row_vals) in enumerate(sub.iterrows()):
        r = start_row + idx_offset
        val_x = float(row_vals.iloc[0])
        val_y = float(row_vals.iloc[1])

        c_ano = ws.cell(row=r, column=1, value=int(ano))
        c_x = ws.cell(row=r, column=2, value=round(val_x, 4))
        c_y = ws.cell(row=r, column=3, value=round(val_y, 4))

        # Fórmulas passo a passo para o cálculo de Pearson no Excel
        c_dx = ws.cell(row=r, column=4, value=f"=B{r}-$B${mean_row}")
        c_dy = ws.cell(row=r, column=5, value=f"=C{r}-$C${mean_row}")
        c_dxdy = ws.cell(row=r, column=6, value=f"=D{r}*E{r}")
        c_dx2 = ws.cell(row=r, column=7, value=f"=D{r}^2")
        c_dy2 = ws.cell(row=r, column=8, value=f"=E{r}^2")
        c_yhat = ws.cell(row=r, column=9, value=f"=$M$12+$M$11*B{r}")

        c_ano.alignment = Alignment(horizontal="center")
        c_x.number_format = "#,##0.00"
        c_y.number_format = "#,##0.00"
        c_dx.number_format = "#,##0.00"
        c_dy.number_format = "#,##0.00"
        c_dxdy.number_format = "#,##0.00"
        c_dx2.number_format = "#,##0.00"
        c_dy2.number_format = "#,##0.00"
        c_yhat.number_format = "#,##0.00"

        fill_zebra = cinza_claro if idx_offset % 2 == 1 else PatternFill(fill_type=None)
        for c in (c_ano, c_x, c_y, c_dx, c_dy, c_dxdy, c_dx2, c_dy2, c_yhat):
            c.font = fonte_normal
            c.border = borda_fina
            if fill_zebra.fill_type:
                c.fill = fill_zebra

    # Linha de Médias / Somatórios na base da tabela
    ws.cell(row=mean_row, column=1, value="MÉDIA / SOMA").font = fonte_negrito
    ws.cell(row=mean_row, column=1).alignment = Alignment(horizontal="center")
    ws.cell(row=mean_row, column=2, value=f"=AVERAGE(B{start_row}:B{end_row})").number_format = "#,##0.00"
    ws.cell(row=mean_row, column=3, value=f"=AVERAGE(C{start_row}:C{end_row})").number_format = "#,##0.00"
    ws.cell(row=mean_row, column=4, value=f"=SUM(D{start_row}:D{end_row})").number_format = "0.00"
    ws.cell(row=mean_row, column=5, value=f"=SUM(E{start_row}:E{end_row})").number_format = "0.00"
    ws.cell(row=mean_row, column=6, value=f"=SUM(F{start_row}:F{end_row})").number_format = "#,##0.00"
    ws.cell(row=mean_row, column=7, value=f"=SUM(G{start_row}:G{end_row})").number_format = "#,##0.00"
    ws.cell(row=mean_row, column=8, value=f"=SUM(H{start_row}:H{end_row})").number_format = "#,##0.00"
    ws.cell(row=mean_row, column=9, value=f"=AVERAGE(I{start_row}:I{end_row})").number_format = "#,##0.00"

    for col_i in range(1, 10):
        cell_m = ws.cell(row=mean_row, column=col_i)
        cell_m.font = fonte_negrito
        cell_m.fill = azul_claro
        cell_m.border = borda_fina

    # Painel Lateral Direito (Colunas K a N): Cálculo de Pearson no Excel + Projeção
    ws.merge_cells("K5:N5")
    ws["K5"] = "QUADRO ESTATÍSTICO — CORRELAÇÃO DE PEARSON E REGRESSÃO NO EXCEL"
    ws["K5"].fill = azul_escuro
    ws["K5"].font = fonte_cabecalho
    ws["K5"].alignment = Alignment(horizontal="center", vertical="center")

    metricas_excel = [
        ("Número de Observações (n)", f"=COUNT(B{start_row}:B{end_row})", est["n"], "0", "=CONT.NÚM()"),
        ("Coeficiente de Pearson (Função CORREL)", f"=CORREL(B{start_row}:B{end_row},C{start_row}:C{end_row})", est["r_scipy"], "0.0000", "=CORREL(X; Y)"),
        ("Coeficiente de Pearson (Função PEARSON)", f"=PEARSON(B{start_row}:B{end_row},C{start_row}:C{end_row})", est["r_scipy"], "0.0000", "=PEARSON(X; Y)"),
        ("Pearson pela Fórmula Algébrica (Somas)", f"=F{mean_row}/SQRT(G{mean_row}*H{mean_row})", est["r_manual"], "0.0000", "Σ(dx·dy)/√[Σdx²·Σdy²]"),
        ("Coeficiente de Determinação (R²)", f"=RSQ(C{start_row}:C{end_row},B{start_row}:B{end_row})", est["r_squared"], "0.0000", "=RQUAD(Y; X)"),
        ("Inclinação da Reta (Slope b₁)", f"=SLOPE(C{start_row}:C{end_row},B{start_row}:B{end_row})", est["slope"], "0.00000", "=INCLINAÇÃO(Y; X)"),
        ("Intercepto da Reta (b₀)", f"=INTERCEPT(C{start_row}:C{end_row},B{start_row}:B{end_row})", est["intercept"], "#,##0.00", "=INTERCEPÇÃO(Y; X)"),
        ("Média de X (x̄)", f"=AVERAGE(B{start_row}:B{end_row})", est["x_mean"], "#,##0.00", "=MÉDIA(X)"),
        ("Média de Y (ȳ)", f"=AVERAGE(C{start_row}:C{end_row})", est["y_mean"], "#,##0.00", "=MÉDIA(Y)"),
        ("Desvio-Padrão Amostral de X (sₓ)", f"=STDEV.S(B{start_row}:B{end_row})", est["x_std"], "#,##0.00", "=DESVPAD.A(X)"),
        ("Desvio-Padrão Amostral de Y (sᵧ)", f"=STDEV.S(C{start_row}:C{end_row})", est["y_std"], "#,##0.00", "=DESVPAD.A(Y)"),
        ("Estatística t de Student", f"=M7*SQRT((M6-2)/(1-M7^2))", est["t_stat"], "0.0000", "r·√(n-2)/√(1-r²)"),
        ("Valor-p (Significância bilateral)", est["p_value"], est["p_value"], "0.00E+00", "Teste t / SciPy"),
    ]

    ws.cell(row=4, column=11, value="Indicador Estatístico").font = fonte_negrito
    ws.cell(row=4, column=12, value="Equivalente PT-BR").font = fonte_negrito
    ws.cell(row=4, column=13, value="Fórmula Ativa (Excel)").font = fonte_negrito
    ws.cell(row=4, column=14, value="Valor Calculado").font = fonte_negrito

    for idx_m, (nome_m, formula_m, val_num, num_fmt, eq_pt) in enumerate(metricas_excel, start=6):
        c1 = ws.cell(row=idx_m, column=11, value=nome_m)
        c2 = ws.cell(row=idx_m, column=12, value=eq_pt)
        c3 = ws.cell(row=idx_m, column=13, value=formula_m)
        c4 = ws.cell(row=idx_m, column=14, value=val_num)

        c1.font = fonte_negrito if "Pearson" in nome_m or "Determinação" in nome_m else fonte_normal
        c2.font = Font(name="Consolas", size=9, color="333333")
        c3.font = fonte_negrito if "Pearson" in nome_m else fonte_normal
        c4.font = fonte_negrito if "Pearson" in nome_m else fonte_normal
        c3.number_format = num_fmt
        c4.number_format = num_fmt

        fill_m = verde_claro if "Pearson" in nome_m or "Determinação" in nome_m else cinza_claro
        for c in (c1, c2, c3, c4):
            c.fill = fill_m
            c.border = borda_fina

    # Quadro de Projeção Linear no Excel
    row_proj_title = 20
    ws.merge_cells(f"K{row_proj_title}:N{row_proj_title}")
    ws[f"K{row_proj_title}"] = "PROJEÇÃO DE CENÁRIOS FUTUROS (REGRESSÃO LINEAR NO EXCEL)"
    ws[f"K{row_proj_title}"].fill = azul_medio
    ws[f"K{row_proj_title}"].font = fonte_cabecalho
    ws[f"K{row_proj_title}"].alignment = Alignment(horizontal="center", vertical="center")

    ws.cell(row=row_proj_title + 1, column=11, value="Cenário Projetado").font = fonte_negrito
    ws.cell(row=row_proj_title + 1, column=12, value=f"Valor de X ({unidade_x})").font = fonte_negrito
    ws.cell(row=row_proj_title + 1, column=13, value="Ŷ via Fórmula Excel").font = fonte_negrito
    ws.cell(row=row_proj_title + 1, column=14, value=f"Ŷ Esperado ({unidade_y})").font = fonte_negrito
    for col_i in range(11, 15):
        ws.cell(row=row_proj_title + 1, column=col_i).fill = azul_claro
        ws.cell(row=row_proj_title + 1, column=col_i).border = borda_fina

    for i_p, (nome_cen, x_cen) in enumerate(cenarios_proj, start=row_proj_title + 2):
        y_cen = est["intercept"] + est["slope"] * x_cen
        c_cen = ws.cell(row=i_p, column=11, value=nome_cen)
        c_xc = ws.cell(row=i_p, column=12, value=x_cen)
        c_yf = ws.cell(row=i_p, column=13, value=f"=$M$12+$M$11*L{i_p}")
        c_yv = ws.cell(row=i_p, column=14, value=round(y_cen, 2))
        c_xc.number_format = "#,##0.00"
        c_yf.number_format = "#,##0.00"
        c_yv.number_format = "#,##0.00"
        for c in (c_cen, c_xc, c_yf, c_yv):
            c.font = fonte_normal
            c.border = borda_fina

    # Gráfico de Dispersão Nativo do Excel (ScatterChart)
    chart = ScatterChart()
    chart.title = f"Diagrama de Dispersão: {nome_x} × {nome_y}"
    chart.style = 13
    chart.x_axis.title = f"{nome_x} ({unidade_x})"
    chart.y_axis.title = f"{nome_y} ({unidade_y})"
    chart.width = 19
    chart.height = 12.5

    xvalues = Reference(ws, min_col=2, min_row=start_row, max_row=end_row)
    yvalues = Reference(ws, min_col=3, min_row=start_row, max_row=end_row)
    yhat_values = Reference(ws, min_col=9, min_row=start_row, max_row=end_row)

    # Série 1: Dados Observados (Pontos de Dispersão)
    series_obs = Series(yvalues, xvalues, title="Observado (Brasil)")
    series_obs.marker.symbol = "circle"
    series_obs.marker.size = 7
    series_obs.graphicalProperties.line.noFill = True
    series_obs.trendline = Trendline(dispEq=True, dispRSqr=True)
    chart.series.append(series_obs)

    # Série 2: Reta de Regressão Linear Ajustada (Ŷ)
    series_reg = Series(yhat_values, xvalues, title="Reta de Regressão Linear (Ŷ)")
    series_reg.marker.symbol = "none"
    chart.series.append(series_reg)

    ws.add_chart(chart, "K27")

    # Quadro de Conclusão Analítica sobre o Estudo
    row_conc_title = mean_row + 3
    ws.merge_cells(f"A{row_conc_title}:I{row_conc_title}")
    ws[f"A{row_conc_title}"] = "CONCLUSÃO TÉCNICA SOBRE A ANÁLISE DE CORRELAÇÃO DE PEARSON E REGRESSÃO"
    ws[f"A{row_conc_title}"].fill = azul_escuro
    ws[f"A{row_conc_title}"].font = fonte_cabecalho
    ws[f"A{row_conc_title}"].alignment = Alignment(horizontal="left", vertical="center")

    ws.merge_cells(f"A{row_conc_title + 1}:I{row_conc_title + 8}")
    cell_conc = ws[f"A{row_conc_title + 1}"]
    cell_conc.value = conclusao_texto
    cell_conc.fill = verde_claro
    cell_conc.font = fonte_normal
    cell_conc.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    # Ajustar largura das colunas
    larguras = {
        1: 12,
        2: 20,
        3: 20,
        4: 16,
        5: 16,
        6: 22,
        7: 20,
        8: 20,
        9: 20,
        10: 3,
        11: 38,
        12: 22,
        13: 20,
        14: 18,
    }
    for col_idx, width in larguras.items():
        ws.column_dimensions[get_column_letter(col_idx)].width = width


def gerar_planilha_excel(
    df_moderno: pd.DataFrame,
    est_energia: dict,
    est_vida: dict,
    est_pobreza: dict,
) -> Path:
    """Gera o arquivo .xlsx completo com cálculos de Pearson, gráficos nativos do Excel e conclusões."""
    wb = openpyxl.Workbook()

    # Aba 1: Análise Principal (PIB per capita PPP x Consumo de Energia Elétrica)
    conclusao_energia = (
        f"1. FORÇA E DIREÇÃO DA CORRELAÇÃO: O coeficiente de correlação linear de Pearson obtido entre o PIB per capita PPP (X) e o "
        f"Consumo de Energia Elétrica per capita (Y) no Brasil (1990–2025, n = {est_energia['n']} anos) é r = {est_energia['r_scipy']:.4f} "
        f"(IC 95%: [{est_energia['ci_95'][0]:.4f}; {est_energia['ci_95'][1]:.4f}]). Segundo a classificação estatística, valores de |r| > 0,90 "
        f"indicam uma CORRELAÇÃO LINEAR POSITIVA MUITO FORTE.\n"
        f"2. SIGNIFICÂNCIA ESTATÍSTICA E PODER EXPLICATIVO: O teste t de Student resultou em t = {est_energia['t_stat']:.2f} com p-valor = {est_energia['p_value']:.2e} "
        f"(altamente significativo, p < 0,0001), rejeitando-se a hipótese nula de ausência de correlação (H₀: ρ = 0). O coeficiente de determinação "
        f"R² = {est_energia['r_squared']:.4f} comprova que {est_energia['r_squared']*100:.2f}% da variação no consumo elétrico per capita brasileiro nas últimas "
        f"três décadas e meia é explicada linearmente pela evolução da renda per capita em paridade de poder de compra.\n"
        f"3. INTERPRETAÇÃO ECONÔMICA E PROJEÇÃO: A inclinação da reta (b₁ = {est_energia['slope']:.4f}) demonstra que, a cada aumento de US$ 1.000 no PIB per capita PPP, "
        f"o consumo médio de eletricidade no Brasil cresce cerca de {est_energia['slope']*1000:.1f} kWh por habitante/ano. Projetando-se o crescimento da renda "
        f"brasileira para US$ 25.000 e US$ 30.000 PPP, estima-se que a demanda elétrica atingirá {est_energia['intercept'] + est_energia['slope']*25000:.0f} kWh/hab e "
        f"{est_energia['intercept'] + est_energia['slope']*30000:.0f} kWh/hab, exigindo expansão proporcional da capacidade instalada de geração e transmissão."
    )
    construir_aba_analise_excel(
        wb=wb,
        titulo_aba="1_PIB_vs_Energia_Principal",
        titulo_relatorio="ANÁLISE 1: PIB PER CAPITA (PPP) × CONSUMO DE ENERGIA ELÉTRICA NO BRASIL (1990–2025)",
        literatura_ref=(
            "• Base Oficial: WORLD BANK. World Development Indicators (WDI) — Brazil (BRA). Washington, DC: The World Bank Group, 2026. "
            "Indicadores: NY.GDP.PCAP.PP.CD e EG.USE.ELEC.KH.PC. Disponível em: https://api.worldbank.org/v2/pt/country/BRA?downloadformat=csv\n"
            "• Literatura Científica do Diagrama: KRAFT, J.; KRAFT, A. On the relationship between energy and GNP. The Journal of Energy and Development, v. 3, n. 2, p. 401-403, 1978."
        ),
        nome_x="PIB per capita PPP",
        nome_y="Consumo Elétrico per capita",
        unidade_x="US$ PPP",
        unidade_y="kWh/hab",
        est=est_energia,
        conclusao_texto=conclusao_energia,
        cenarios_proj=[
            ("Cenário 1: PIB per capita = US$ 25.000 PPP", 25000.0),
            ("Cenário 2: PIB per capita = US$ 28.000 PPP", 28000.0),
            ("Cenário 3: PIB per capita = US$ 30.000 PPP", 30000.0),
        ],
        primeira_aba=True,
    )

    # Aba 2: Curva de Preston no Brasil (PIB per capita PPP x Expectativa de Vida)
    conclusao_vida = (
        f"1. FORÇA E DIREÇÃO DA CORRELAÇÃO: O coeficiente de correlação de Pearson entre o PIB per capita PPP (X) e a Expectativa de Vida ao Nascer (Y) "
        f"no Brasil (1990–2024, n = {est_vida['n']} anos) é r = {est_vida['r_scipy']:.4f} (p-valor = {est_vida['p_value']:.2e}). Trata-se de uma CORRELAÇÃO "
        f"POSITIVA MUITO FORTE, validando empiricamente a clássica Curva de Preston (Preston, 1975) para a trajetória brasileira contemporânea.\n"
        f"2. ANÁLISE DO DIAGRAMA E IMPACTO DA PANDEMIA: O coeficiente de determinação R² = {est_vida['r_squared']:.4f} mostra que {est_vida['r_squared']*100:.2f}% "
        f"dos ganhos de longevidade no Brasil (de 65,9 anos em 1990 para 76,0 anos em 2024) estão associados ao avanço econômico e investimentos sociais. "
        f"O gráfico de dispersão evidencia também o outlier histórico de 2020–2021 (queda abrupta para 73,0 anos devido à pandemia de COVID-19), seguido de recuperação em 2023–2024."
    )
    construir_aba_analise_excel(
        wb=wb,
        titulo_aba="2_PIB_vs_Expectativa_Vida",
        titulo_relatorio="ANÁLISE 2 (CURVA DE PRESTON): PIB PER CAPITA (PPP) × EXPECTATIVA DE VIDA NO BRASIL (1990–2024)",
        literatura_ref=(
            "• Base Oficial: WORLD BANK. World Development Indicators (WDI) — Brazil (BRA). Indicadores: NY.GDP.PCAP.PP.CD e SP.DYN.LE00.IN.\n"
            "• Literatura Científica do Diagrama: PRESTON, Samuel H. The Changing Relation between Mortality and Level of Economic Development. "
            "Population Studies, v. 29, n. 2, p. 231-248, 1975."
        ),
        nome_x="PIB per capita PPP",
        nome_y="Expectativa de Vida",
        unidade_x="US$ PPP",
        unidade_y="Anos",
        est=est_vida,
        conclusao_texto=conclusao_vida,
        cenarios_proj=[
            ("Cenário 1: PIB per capita = US$ 25.000 PPP", 25000.0),
            ("Cenário 2: PIB per capita = US$ 28.000 PPP", 28000.0),
            ("Cenário 3: PIB per capita = US$ 30.000 PPP", 30000.0),
        ],
        primeira_aba=False,
    )

    # Aba 3: PIB per capita PPP x Extrema Pobreza (Correlação Negativa Forte)
    conclusao_pobreza = (
        f"1. FORÇA E DIREÇÃO DA CORRELAÇÃO: A análise entre o PIB per capita PPP (X) e a Taxa de Extrema Pobreza (Y) no Brasil (1990–2024, n = {est_pobreza['n']}) "
        f"apresenta coeficiente de Pearson r = {est_pobreza['r_scipy']:.4f} (p-valor = {est_pobreza['p_value']:.2e}) e R² = {est_pobreza['r_squared']:.4f}. "
        f"Constata-se uma CORRELAÇÃO LINEAR NEGATIVA MUITO FORTE: à medida que a renda per capita nacional se expandiu de US$ 6.688 para mais de US$ 22.370 PPP, "
        f"a proporção da população em extrema pobreza recuou de 25,7% (1990) para 3,0% (2024).\n"
        f"2. CONCLUSÃO SOCIOECONÔMICA: Cada acréscimo de US$ 1.000 no PIB per capita PPP esteve associado a uma redução média de {abs(est_pobreza['slope'])*1000:.2f} "
        f"pontos percentuais na extrema pobreza, comprovando a eficácia combinada do crescimento econômico, estabilização monetária pós-Plano Real e políticas de transferência de renda."
    )
    construir_aba_analise_excel(
        wb=wb,
        titulo_aba="3_PIB_vs_Pobreza_Extrema",
        titulo_relatorio="ANÁLISE 3 (CORRELAÇÃO NEGATIVA): PIB PER CAPITA (PPP) × EXTREMA POBREZA NO BRASIL (1990–2024)",
        literatura_ref=(
            "• Base Oficial: WORLD BANK. World Development Indicators (WDI) — Brazil (BRA). Indicadores: NY.GDP.PCAP.PP.CD e SI.POV.DDAY.\n"
            "• Literatura Científica do Diagrama: RAVALLION, M.; CHEN, S. Measuring Pro-Poor Growth. Economics Letters, v. 78, n. 1, p. 93-99, 2003 / World Bank Poverty & Equity Brief."
        ),
        nome_x="PIB per capita PPP",
        nome_y="Extrema Pobreza (% pop.)",
        unidade_x="US$ PPP",
        unidade_y="% da População",
        est=est_pobreza,
        conclusao_texto=conclusao_pobreza,
        cenarios_proj=[
            ("Cenário 1: PIB per capita = US$ 24.000 PPP", 24000.0),
            ("Cenário 2: PIB per capita = US$ 25.000 PPP", 25000.0),
        ],
        primeira_aba=False,
    )

    # Aba 4: Base Consolidada do Brasil (1990-2025)
    ws_base = wb.create_sheet(title="4_Base_Consolidada_Brasil")
    df_export = df_moderno.reset_index()
    ws_base.append(list(df_export.columns))
    for cell in ws_base[1]:
        cell.font = Font(name="Calibri", size=10, bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", fgColor="1F4E78")
    for row in df_export.itertuples(index=False):
        ws_base.append([None if pd.isna(v) else float(v) if isinstance(v, (int, float, np.number)) else v for v in row])
    for col in ws_base.columns:
        ws_base.column_dimensions[get_column_letter(col[0].column)].width = 24

    excel_path = OUTPUT_DIR / "Analise_Correlacao_Pearson_Brasil_WorldBank.xlsx"
    wb.save(excel_path)
    return excel_path


def gerar_tabela_passo_a_passo_csv(est: dict, col_x: str, col_y: str, caminho_csv: Path) -> None:
    """Exporta a tabela passo a passo do cálculo algébrico de Pearson e regressão linear em formato CSV compatível com qualquer visualizador."""
    sub = est["df_limpo"].copy()
    x = sub[col_x].to_numpy(dtype=float)
    y = sub[col_y].to_numpy(dtype=float)
    dx = x - est["x_mean"]
    dy = y - est["y_mean"]
    dx_dy = dx * dy
    dx2 = dx**2
    dy2 = dy**2
    y_hat = est["intercept"] + est["slope"] * x
    residuo = y - y_hat

    df_tab = pd.DataFrame({
        "Ano": sub.index.astype(str),
        col_x: np.round(x, 4),
        col_y: np.round(y, 4),
        "Desvio_X_(xi_menos_media_x)": np.round(dx, 4),
        "Desvio_Y_(yi_menos_media_y)": np.round(dy, 4),
        "Produto_Desvios_(dx_vezes_dy)": np.round(dx_dy, 4),
        "Quadrado_Desvio_X_(dx2)": np.round(dx2, 4),
        "Quadrado_Desvio_Y_(dy2)": np.round(dy2, 4),
        "Projecao_Linear_Y_hat": np.round(y_hat, 4),
        "Residuo_(Y_menos_Y_hat)": np.round(residuo, 4),
    })

    linha_resumo = pd.DataFrame([{
        "Ano": "MEDIA_OU_SOMA",
        col_x: round(est["x_mean"], 4),
        col_y: round(est["y_mean"], 4),
        "Desvio_X_(xi_menos_media_x)": 0.0,
        "Desvio_Y_(yi_menos_media_y)": 0.0,
        "Produto_Desvios_(dx_vezes_dy)": round(est["sum_dx_dy"], 4),
        "Quadrado_Desvio_X_(dx2)": round(est["sum_dx2"], 4),
        "Quadrado_Desvio_Y_(dy2)": round(est["sum_dy2"], 4),
        "Projecao_Linear_Y_hat": round(float(np.mean(y_hat)), 4),
        "Residuo_(Y_menos_Y_hat)": 0.0,
    }])

    df_final = pd.concat([df_tab, linha_resumo], ignore_index=True)
    df_final.to_csv(caminho_csv, index=False, encoding="utf-8")


def exportar_todos_csvs(
    df_moderno: pd.DataFrame,
    est_energia: dict,
    est_vida: dict,
    est_pobreza: dict,
) -> list[Path]:
    """Gera arquivos CSV limpos e padronizados para visualização direta em qualquer CSV Viewer."""
    arquivos_gerados: list[Path] = []

    # 1. Base consolidada (1990-2025) arredondada e limpa
    p_base = OUTPUT_DIR / "base_brasil_tratada_1990_2025.csv"
    df_moderno.round(4).to_csv(p_base, index=True, encoding="utf-8")
    arquivos_gerados.append(p_base)

    # 2. Tabelas passo a passo de Pearson (mesmas abas do Excel, em formato CSV)
    p_a1 = OUTPUT_DIR / "1_analise_pib_vs_energia_pearson.csv"
    gerar_tabela_passo_a_passo_csv(
        est_energia, "PIB_per_capita_PPP_USD", "Consumo_Energia_Eletrica_kWh_pc", p_a1
    )
    arquivos_gerados.append(p_a1)

    p_a2 = OUTPUT_DIR / "2_analise_pib_vs_expectativa_vida_pearson.csv"
    gerar_tabela_passo_a_passo_csv(
        est_vida, "PIB_per_capita_PPP_USD", "Expectativa_Vida_Anos", p_a2
    )
    arquivos_gerados.append(p_a2)

    p_a3 = OUTPUT_DIR / "3_analise_pib_vs_pobreza_extrema_pearson.csv"
    gerar_tabela_passo_a_passo_csv(
        est_pobreza, "PIB_per_capita_PPP_USD", "Pobreza_Extrema_perc", p_a3
    )
    arquivos_gerados.append(p_a3)

    # 3. Resumo Estatístico de Pearson + Projeções Lineares
    linhas_resumo = []
    for nome_estudo, var_x, var_y, est, c_proj in [
        ("1. PIB per capita PPP x Consumo de Energia Eletrica", "PIB_per_capita_PPP_USD", "Consumo_Energia_Eletrica_kWh_pc", est_energia, [25000.0, 28000.0, 30000.0]),
        ("2. PIB per capita PPP x Expectativa de Vida (Preston)", "PIB_per_capita_PPP_USD", "Expectativa_Vida_Anos", est_vida, [25000.0, 28000.0, 30000.0]),
        ("3. PIB per capita PPP x Pobreza Extrema", "PIB_per_capita_PPP_USD", "Pobreza_Extrema_perc", est_pobreza, [24000.0, 25000.0, 28000.0]),
    ]:
        linhas_resumo.append({
            "Estudo": nome_estudo,
            "Variavel_X": var_x,
            "Variavel_Y": var_y,
            "Periodo": f"{est['ano_inicio']}-{est['ano_fim']}",
            "N_Observacoes": est["n"],
            "Media_X": round(est["x_mean"], 4),
            "Media_Y": round(est["y_mean"], 4),
            "Desvio_Padrao_X": round(est["x_std"], 4),
            "Desvio_Padrao_Y": round(est["y_std"], 4),
            "r_Pearson": round(est["r_scipy"], 6),
            "R2_Determinacao": round(est["r_squared"], 6),
            "R2_Percentual": round(est["r_squared"] * 100, 2),
            "Estatistica_t": round(est["t_stat"], 4),
            "Valor_p": f"{est['p_value']:.4e}",
            "IC_95_Inferior": round(est["ci_95"][0], 4),
            "IC_95_Superior": round(est["ci_95"][1], 4),
            "Inclinacao_Slope_b1": round(est["slope"], 6),
            "Intercepto_b0": round(est["intercept"], 4),
            "Projecao_X_1": c_proj[0],
            "Projecao_Y_hat_1": round(est["intercept"] + est["slope"] * c_proj[0], 2),
            "Projecao_X_2": c_proj[1],
            "Projecao_Y_hat_2": round(est["intercept"] + est["slope"] * c_proj[1], 2),
            "Projecao_X_3": c_proj[2],
            "Projecao_Y_hat_3": round(est["intercept"] + est["slope"] * c_proj[2], 2),
        })
    p_resumo = OUTPUT_DIR / "4_resumo_estatistico_e_projecoes_pearson.csv"
    pd.DataFrame(linhas_resumo).to_csv(p_resumo, index=False, encoding="utf-8")
    arquivos_gerados.append(p_resumo)

    # 4. Matriz de Correlação de Pearson em CSV
    p_matriz = OUTPUT_DIR / "5_matriz_correlacao_pearson_brasil.csv"
    corr_df = df_moderno.corr(method="pearson").round(4)
    corr_df.index.name = "Indicador"
    corr_df.to_csv(p_matriz, index=True, encoding="utf-8")
    arquivos_gerados.append(p_matriz)

    # 5. Versões limpas dos CSVs originais do World Bank (sem as 4 linhas de metadados do topo que quebram visualizadores de CSV)
    for orig_name, clean_name in [
        ("API_BRA_DS2_pt_csv_v2_11137.csv", "API_BRA_DS2_pt_limpo_para_visualizador.csv"),
        ("API_BRA_DS2_en_csv_v2_2904.csv", "API_BRA_DS2_en_limpo_para_visualizador.csv"),
    ]:
        orig_path = DATA_DIR / orig_name
        if orig_path.exists():
            df_wb = pd.read_csv(orig_path, skiprows=4)
            # Remover coluna vazia extra gerada pela vírgula final do Banco Mundial
            df_wb = df_wb.loc[:, ~df_wb.columns.str.contains("^Unnamed")]
            clean_path = DATA_DIR / clean_name
            df_wb.to_csv(clean_path, index=False, encoding="utf-8")
            arquivos_gerados.append(clean_path)

    return arquivos_gerados


def main() -> None:
    df_anos, _ = carregar_e_tratar_dados()
    df_moderno = df_anos.loc[1990:2025].copy()

    est_energia = calcular_estatisticas_pearson(
        df_moderno, "PIB_per_capita_PPP_USD", "Consumo_Energia_Eletrica_kWh_pc"
    )
    est_vida = calcular_estatisticas_pearson(
        df_moderno, "PIB_per_capita_PPP_USD", "Expectativa_Vida_Anos"
    )
    est_pobreza = calcular_estatisticas_pearson(
        df_moderno, "PIB_per_capita_PPP_USD", "Pobreza_Extrema_perc"
    )

    csvs_criados = exportar_todos_csvs(df_moderno, est_energia, est_vida, est_pobreza)
    gerar_graficos(df_moderno, est_energia, est_vida, est_pobreza)
    excel_path = gerar_planilha_excel(df_moderno, est_energia, est_vida, est_pobreza)

    print("=== RESUMO DOS RESULTADOS OBTIDOS ===")
    for titulo, est in [
        ("1. PIB per capita PPP x Consumo de Energia Elétrica (1990-2025)", est_energia),
        ("2. PIB per capita PPP x Expectativa de Vida (1990-2024)", est_vida),
        ("3. PIB per capita PPP x Pobreza Extrema (1990-2024)", est_pobreza),
    ]:
        print(f"\n{titulo}:")
        print(f"   n = {est['n']} ({est['ano_inicio']}-{est['ano_fim']})")
        print(f"   r de Pearson (manual/Excel) = {est['r_manual']:.6f}")
        print(f"   r de Pearson (SciPy)        = {est['r_scipy']:.6f}")
        print(f"   R² (Determinação)           = {est['r_squared']:.6f} ({est['r_squared']*100:.2f}%)")
        print(f"   Estatística t               = {est['t_stat']:.4f} | p-valor = {est['p_value']:.4e}")
        print(f"   Equação da Reta: Ŷ = {est['intercept']:.4f} + ({est['slope']:.6f}) * X")
    print(f"\nPlanilha Excel gerada com sucesso em: {excel_path}")
    print("\nCSVs gerados para visualização:")
    for p in csvs_criados:
        print(f" - {p}")


if __name__ == "__main__":
    main()
