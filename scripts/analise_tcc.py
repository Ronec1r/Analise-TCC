# -----------------------------------------------------------------------------
# Script de Análise de Dados - TCC: Impacto do VisuAlgo no Ensino de Árvores
# Autor: Rone Clay Oliveira Andrade
# Data: Dezembro/2025
# Descrição: Processamento dos dados brutos do experimento (Pós-teste, SUS, NPS)
#            e geração automática dos gráficos para o Capítulo 5.
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuração de Estilo Global
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Cria pasta para salvar os resultados se não existir
if not os.path.exists('resultados'):
    os.makedirs('resultados')

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS (DADOS BRUTOS)
# -----------------------------------------------------------------------------
# Estrutura: ID, Grupo, Pre, Pos, Q1..Q14 (0=Erro, 1=Acerto)
raw_data = [
    [1, 'controle', 7, 13, 1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [2, 'controle', 7, 12, 1,1,1,1,1,1,1,1,1,0,0,1,1,1],
    [3, 'controle', 13, 13, 1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1][:18], # Ajuste de tamanho
    [4, 'controle', 6, 11, 1,1,1,1,0,0,1,0,1,1,1,1,1,1],
    [5, 'controle', 7, 10, 1,0,1,1,1,1,1,0,1,0,0,1,1,1],
    [6, 'experimental', 5, 10, 1,0,1,1,1,1,0,0,1,1,1,1,1,0], # Aluno A1
    [7, 'experimental', 10, 9, 1,1,1,1,1,0,0,0,1,0,0,1,1,1], # Aluno A2
    [8, 'experimental', 6, 8, 1,1,1,1,1,1,0,0,1,0,0,0,1,0],  # Aluno A3
    [9, 'experimental', 7, 10, 1,1,1,1,1,0,0,1,1,0,0,1,1,1], # Aluno A4
    [10, 'experimental', 7, 12, 1,1,1,1,0,0,1,1,1,1,1,1,1,1], # Aluno A5
    [11, 'experimental', 6, 11, 1,1,1,1,0,0,1,0,1,1,1,1,1,1]  # Aluno A6
]

cols = ['id', 'grupo', 'pre', 'pos'] + [f'q{i}' for i in range(1, 15)]
df = pd.DataFrame(raw_data, columns=cols)

# Dados SUS (System Usability Scale) - Apenas Experimental
# IDs correspondem a: 6->A1, 7->A2, 8->A3, 9->A4, 10->A5, 11->A6
sus_data = {
    'aluno': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'],
    'score': [72.5, 65.0, 47.5, 92.5, 77.5, 87.5],
    'nps_nota': [8, 7, 7, 10, 9, 10] # Inferido da classificação NPS
}
df_sus = pd.DataFrame(sus_data)

# Mapeamento de Questões
cat_fundamentos = ['q1', 'q2', 'q3', 'q4']
cat_percursos = ['q5', 'q6', 'q7']
cat_operacoes = ['q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14']

print(">>> Dados carregados com sucesso.")

# -----------------------------------------------------------------------------
# 2. GERAÇÃO FIGURA 1: COMPARAÇÃO GERAL (MÉDIAS)
# -----------------------------------------------------------------------------
media_controle = df[df['grupo'] == 'controle']['pos'].mean()
media_exp = df[df['grupo'] == 'experimental']['pos'].mean()
std_controle = df[df['grupo'] == 'controle']['pos'].std()
std_exp = df[df['grupo'] == 'experimental']['pos'].std()

grupos = ['Controle\n(Tradicional)', 'Experimental\n(VisuAlgo)']
medias = [media_controle, media_exp]
erros = [std_controle, std_exp]

plt.figure(figsize=(8, 6))
bars = plt.bar(grupos, medias, yerr=erros, capsize=10, color=['#4c72b0', '#dd8452'], alpha=0.9)
plt.ylabel('Nota Média (0-14)')
plt.title('Comparativo de Desempenho Geral (Pós-Teste)')
plt.ylim(0, 15)

# Adicionar valores nas barras
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 1.5,
             f'{height:.2f}', ha='center', va='bottom', color='white', fontweight='bold')

plt.savefig('resultados/Figura_1_Comparacao_Geral.png', dpi=300)
print(">>> Figura 1 salva.")

# -----------------------------------------------------------------------------
# 3. GERAÇÃO FIGURA 2: DESEMPENHO POR CATEGORIA
# -----------------------------------------------------------------------------
res = {}
for g in ['controle', 'experimental']:
    sub = df[df['grupo'] == g]
    res[g] = [
        sub[cat_fundamentos].values.mean() * 100,
        sub[cat_percursos].values.mean() * 100,
        sub[cat_operacoes].values.mean() * 100
    ]

labels = ['Fundamentos\n(Visual/Estático)', 'Percursos\n(Recursão/Lógica)', 'Operações\n(Regras Dinâmicas)']
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, res['controle'], width, label='Controle', color='#4c72b0')
rects2 = plt.bar(x + width/2, res['experimental'], width, label='Experimental', color='#dd8452')

plt.ylabel('Taxa de Acerto (%)')
plt.title('Desempenho Detalhado por Complexidade Cognitiva')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 115)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.savefig('resultados/Figura_2_Categorias.png', dpi=300)
print(">>> Figura 2 salva.")

# -----------------------------------------------------------------------------
# 4. GERAÇÃO FIGURA 3: SUS (USABILIDADE INDIVIDUAL)
# -----------------------------------------------------------------------------
cores_sus = []
for s in df_sus['score']:
    if s < 50: cores_sus.append('#d62728') # Vermelho
    elif s < 70: cores_sus.append('#ff7f0e') # Laranja
    else: cores_sus.append('#2ca02c') # Verde

plt.figure(figsize=(9, 6))
barras_sus = plt.bar(df_sus['aluno'], df_sus['score'], color=cores_sus, alpha=0.8)

# Linhas de referência
plt.axhline(y=68, color='gray', linestyle='--', label='Média de Mercado (68)')
plt.axhline(y=df_sus['score'].mean(), color='blue', linestyle='-.', label=f'Média Turma ({df_sus["score"].mean():.1f})')

plt.ylabel('Score SUS (0-100)')
plt.title('Avaliação Individual de Usabilidade (SUS)')
plt.ylim(0, 105)
plt.legend(loc='lower left')

for rect in barras_sus:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom', fontweight='bold')

plt.savefig('resultados/Figura_3_SUS.png', dpi=300)
print(">>> Figura 3 salva.")

# -----------------------------------------------------------------------------
# 5. GERAÇÃO FIGURA 4: NPS (SATISFAÇÃO)
# -----------------------------------------------------------------------------
# Cálculo simples: Promotores (9-10), Neutros (7-8), Detratores (0-6)
# Dados baseados na inferência da análise: 3 Promotores, 3 Neutros, 0 Detratores
labels_nps = ['Promotores (9-10)', 'Neutros (7-8)', 'Detratores (0-6)']
sizes_nps = [50, 50, 0] # %
colors_nps = ['#2ca02c', '#ff7f0e', '#d62728']

# Para o gráfico não mostrar a fatia 0%, filtramos
labels_plot = [l for l, s in zip(labels_nps, sizes_nps) if s > 0]
sizes_plot = [s for s in sizes_nps if s > 0]
colors_plot = [c for c, s in zip(colors_nps, sizes_nps) if s > 0]

plt.figure(figsize=(7, 7))
wedges, texts, autotexts = plt.pie(sizes_plot, labels=labels_plot, autopct='%1.1f%%',
                                   startangle=90, colors=colors_plot,
                                   textprops=dict(color="black", fontweight='bold'))

plt.title('Net Promoter Score (NPS = +50)')
plt.savefig('resultados/Figura_4_NPS.png', dpi=300)
print(">>> Figura 4 salva.")

# -----------------------------------------------------------------------------
# 6. EXPORTAÇÃO DE TABELAS FINAIS
# -----------------------------------------------------------------------------
df.to_csv('resultados/tabela_dados_brutos.csv', index=False)
df_sus.to_csv('resultados/tabela_sus_nps.csv', index=False)

print("\n>>> Processamento concluído!")
print(">>> Gráficos e tabelas salvos na pasta 'resultados/'.")
