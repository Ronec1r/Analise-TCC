# -----------------------------------------------------------------------------
# Script de Análise de Dados - TCC: Impacto do VisuAlgo no Ensino de Árvores
# Autor: Rone Clay Oliveira Andrade
# Data: Dezembro/2025
# Descrição: Processamento dos dados brutos ajustados e geração de gráficos.
# -----------------------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuração de Estilo Global
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({'font.size': 12, 'figure.autolayout': True})

# Cria pasta para salvar os resultados
if not os.path.exists('resultados_finais'):
    os.makedirs('resultados_finais')

# -----------------------------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS (DADOS CORRIGIDOS)
# -----------------------------------------------------------------------------
# Mapeamento:
# Controle: IDs 1-5 (A01-A05)
# Experimental: IDs 6-11 (A06-A11) que viram A1-A6 no texto
# Estrutura: [ID, Grupo, Nota_Pre, Nota_Pos, Q1...Q14 (0 ou 1)]

raw_data = [
    # --- GRUPO CONTROLE (Dados da string CSV) ---
    [1, 'controle', 7, 12, 1,1,1,1,1,1,1,1,1,0,0,1,1,1],  # A01
    [2, 'controle', 7, 13, 1,1,1,1,1,1,1,1,1,1,1,1,1,0],  # A02
    [3, 'controle', 6, 8,  1,0,1,1,0,0,1,0,1,0,0,1,1,1],  # A03
    [4, 'controle', 7, 10, 1,0,1,1,1,1,1,0,1,0,0,1,1,1],  # A04
    [5, 'controle', 13, 13, 1,0,1,1,1,1,1,1,1,1,1,1,1,1], # A05
    
    # --- GRUPO EXPERIMENTAL (Dados corrigidos) ---
    # ID 6 = A1 (TCC)
    [6, 'experimental', 5, 10, 1,0,1,1,1,1,0,0,1,1,1,1,1,0], 
    # ID 7 = A2 (TCC)
    [7, 'experimental', 10, 9, 1,1,1,1,1,0,0,0,1,0,0,1,1,1], 
    # ID 8 = A3 (TCC) -> O NOVO ALUNO DE SUCESSO
    [8, 'experimental', 7, 10, 1,1,1,1,1,0,0,1,1,0,0,1,1,1], 
    # ID 9 = A4 (TCC) -> ALTA PERFORMANCE
    [9, 'experimental', 7, 12, 1,1,1,1,0,0,1,1,1,1,1,1,1,1], 
    # ID 10 = A5 (TCC) -> O NOVO ALUNO COM DIFICULDADE (FRICÇÃO)
    [10, 'experimental', 6, 8,  1,1,1,1,1,1,0,0,1,0,0,0,1,0], 
    # ID 11 = A6 (TCC)
    [11, 'experimental', 6, 11, 1,1,1,1,0,0,1,0,1,1,1,1,1,1] 
]

cols = ['id', 'grupo', 'pre', 'pos'] + [f'q{i}' for i in range(1, 15)]
df = pd.DataFrame(raw_data, columns=cols)

# Dados SUS (System Usability Scale) - Recalculados
sus_data = {
    'aluno': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'], # IDs TCC
    'id_real': [6, 7, 8, 9, 10, 11],
    'score': [72.5, 65.0, 92.5, 72.5, 52.5, 87.5], # Valores corrigidos
    'nota_pos': [10, 9, 10, 12, 8, 11] # Para plotar junto se necessário
}
df_sus = pd.DataFrame(sus_data)

# Mapeamento de Categorias de Questões
cat_fundamentos = ['q1', 'q2', 'q3', 'q4']
cat_percursos = ['q5', 'q6', 'q7'] # O ponto fraco do experimental
cat_operacoes = ['q8', 'q9', 'q10', 'q11', 'q12', 'q13', 'q14']

print(">>> Dados carregados e sincronizados com a tabela corrigida.")

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
plt.title(f'Desempenho Geral: Controle ({media_controle:.1f}) vs Exp ({media_exp:.1f})')
plt.ylim(0, 15)

# Adicionar valores
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 2,
             f'{height:.1f}', ha='center', va='bottom', color='white', fontweight='bold', fontsize=14)

plt.savefig('resultados_finais/Figura_1_Comparacao_Geral.png', dpi=300)
print(">>> Figura 1 salva (Controle superior).")

# -----------------------------------------------------------------------------
# 3. GERAÇÃO FIGURA 2: DESEMPENHO POR CATEGORIA (RECURSÃO)
# -----------------------------------------------------------------------------
res = {}
for g in ['controle', 'experimental']:
    sub = df[df['grupo'] == g]
    res[g] = [
        sub[cat_fundamentos].values.mean() * 100,
        sub[cat_percursos].values.mean() * 100,
        sub[cat_operacoes].values.mean() * 100
    ]

labels = ['Fundamentos\n(Estrutura)', 'Percursos\n(Recursão/Pilha)', 'Operações\n(Inserção/Remoção)']
x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
rects1 = plt.bar(x - width/2, res['controle'], width, label='Controle', color='#4c72b0')
rects2 = plt.bar(x + width/2, res['experimental'], width, label='Experimental', color='#dd8452')

plt.ylabel('Taxa de Acerto (%)')
plt.title('Diferença Crítica: O Software falhou em ensinar Recursão')
plt.xticks(x, labels)
plt.legend()
plt.ylim(0, 119)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height:.1f}%',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.savefig('resultados_finais/Figura_2_Categorias.png', dpi=300)
print(">>> Figura 2 salva (Destaque para o gap em Percursos).")

# -----------------------------------------------------------------------------
# 4. GERAÇÃO FIGURA 3: SUS vs NOTA (CORRELAÇÃO)
# -----------------------------------------------------------------------------
# Cores baseadas na classificação SUS
cores_sus = []
for s in df_sus['score']:
    if s < 60: cores_sus.append('#d62728') # Vermelho (Ruim)
    elif s < 80: cores_sus.append('#ff7f0e') # Laranja (Bom)
    else: cores_sus.append('#2ca02c') # Verde (Excelente)

plt.figure(figsize=(10, 6))
barras_sus = plt.bar(df_sus['aluno'], df_sus['score'], color=cores_sus, alpha=0.85)

# Linhas de referência
plt.axhline(y=68, color='gray', linestyle='--', label='Corte de Mercado (68)')

plt.ylabel('Score SUS (0-100)')
plt.title('Usabilidade (SUS) x Desempenho (Nota)')
plt.ylim(0, 110)
plt.legend(loc='upper left')

# Adicionar Nota do Pós-Teste sobre a barra para provar a correlação
for rect, nota in zip(barras_sus, df_sus['nota_pos']):
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/2., height + 2,
             f'SUS: {height}\nNota: {nota}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.savefig('resultados_finais/Figura_3_SUS_Nota.png', dpi=300)
print(">>> Figura 3 salva (A5 baixo, A3 alto).")

# -----------------------------------------------------------------------------
# 5. GERAÇÃO FIGURA 4: NPS (SATISFAÇÃO)
# -----------------------------------------------------------------------------
# NPS Dados Reais: 
# Promotores (9-10): A3(10), A4(10), A6(10) -> 3 alunos
# Neutros (7-8): A1(8), A2(8), A5(7) -> 3 alunos
# Detratores (0-6): 0 alunos

labels_nps = ['Promotores (Notas 9-10)', 'Neutros (Notas 7-8)', 'Detratores']
sizes_nps = [50, 50, 0] 
colors_nps = ['#2ca02c', '#ff7f0e', '#d62728']

# Filtrar zeros
labels_plot = [l for l, s in zip(labels_nps, sizes_nps) if s > 0]
sizes_plot = [s for s in sizes_nps if s > 0]
colors_plot = [c for c, s in zip(colors_nps, sizes_nps) if s > 0]

plt.figure(figsize=(7, 7))
plt.pie(sizes_plot, labels=labels_plot, autopct='%1.1f%%',
        startangle=90, colors=colors_plot,
        textprops=dict(color="black", fontweight='bold', fontsize=12))

plt.title('Net Promoter Score (NPS = +50)')
plt.savefig('resultados_finais/Figura_4_NPS.png', dpi=300)
print(">>> Figura 4 salva.")

# -----------------------------------------------------------------------------
# 6. EXPORTAÇÃO CSV
# -----------------------------------------------------------------------------
df.to_csv('resultados_finais/tabela_dados_completos.csv', index=False)
print("\n>>> Processamento concluído! Verifique a pasta 'resultados_finais'.")
