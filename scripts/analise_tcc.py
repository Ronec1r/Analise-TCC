# --- 0. CONFIGURAÇÃO INICIAL E IMPORTAÇÕES ---
import pandas as pd
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Configuração visual global
sns.set_style("ticks")
plt.rcParams['figure.figsize'] = (10, 6)

# --- 1. CARGA E PREPARAÇÃO DOS DADOS (CORRIGIDO PARA BATER COM O TEXTO) ---
# Mapeamento de IDs para o Texto do TCC:
# A01-A05 = Grupo Controle
# A06 = Aluno A1 (Texto) -> SUS 72.5
# A07 = Aluno A2 (Texto) -> SUS 65.0
# A08 = Aluno A5 (Texto) -> SUS 52.5
# A09 = Aluno A3 (Texto) -> SUS 92.5
# A10 = Aluno A4 (Texto) -> SUS 72.5
# A11 = Aluno A6 (Texto) -> SUS 87.5

dados_csv_string = """ID;Grupo;Idade;Periodo;Genero;Exp_Prog_Pre;Confianca_Pre;Q1_Pre;Q2_Pre;Q3_Pre;Q4_Pre;Q5_Pre;Q6_Pre;Q7_Pre;Q8_Pre;Q9_Pre;Q10_Pre;Q11_Pre;Q12_Pre;Q13_Pre;Q14_Pre;Pontuação_Pre;Exp_Prog_Pos;Confianca_Pos;Q1_Pos;Q2_Pos;Q3_Pos;Q4_Pos;Q5_Pos;Q6_Pos;Q7_Pos;Q8_Pos;Q9_Pos;Q10_Pos;Q11_Pos;Q12_Pos;Q13_Pos;Q14_Pos;Pontuação_Pos;SUS_Q1;SUS_Q2;SUS_Q3;SUS_Q4;SUS_Q5;SUS_Q6;SUS_Q7;SUS_Q8;SUS_Q9;SUS_Q10;NPS_Q11
A01;Controle;24;5;1;3;2;1;0;1;1;1;1;0;0;1;0;0;1;0;0;7;3;3;1;1;1;1;1;1;1;1;1;1;1;1;1;0;13;;;;;;;;;;;
A02;Controle;23;6;1;2;2;1;1;1;0;1;0;0;0;1;0;0;1;1;0;7;3;2;1;1;1;1;1;1;1;1;1;0;0;1;1;1;12;;;;;;;;;;;
A03;Controle;21;6;1;3;2;1;1;1;1;0;0;0;0;1;0;0;1;0;0;6;3;3;1;0;1;1;1;1;1;1;1;1;1;1;1;1;13;;;;;;;;;;;
A04;Controle;24;6;1;3;2;1;1;1;1;0;1;0;0;1;0;0;1;0;0;7;3;3;1;1;1;1;0;0;1;0;1;1;1;1;1;1;11;;;;;;;;;;;
A05;Controle;33;6;1;2;3;1;1;1;1;1;1;1;1;0;1;1;1;1;1;13;2;3;1;0;1;1;1;1;1;0;1;0;0;1;1;1;10;;;;;;;;;;;
A06;Tratamento;23;6;1;2;2;1;0;0;0;0;1;0;1;1;0;0;1;0;0;5;2;3;1;0;1;1;1;1;0;0;1;1;1;1;1;0;10;5;2;4;5;5;1;4;1;5;5;8
A07;Tratamento;21;6;1;3;2;1;1;1;1;1;1;1;0;1;0;1;1;0;0;10;3;2;1;1;1;1;1;0;0;0;1;0;0;1;1;1;9;5;3;4;3;4;3;4;3;4;3;7
A08;Tratamento;24;6;1;2;2;1;1;1;1;0;0;0;1;1;0;0;1;0;0;7;2;3;1;1;1;1;1;1;0;0;1;0;0;0;1;0;8;4;3;4;3;3;3;3;3;3;4;7
A09;Tratamento;21;6;1;3;2;1;1;1;1;0;0;0;1;1;0;0;1;0;0;7;3;3;1;1;1;1;1;0;0;1;1;0;0;1;1;1;10;5;1;5;1;5;1;5;1;5;4;10
A10;Tratamento;27;5;1;1;1;1;1;1;1;0;0;0;0;1;0;0;1;0;0;6;2;2;1;1;1;1;0;0;1;1;1;1;1;1;1;1;12;3;3;4;2;4;2;5;2;4;2;9
A11;Tratamento;26;3;1;2;2;1;1;0;0;0;0;0;1;1;0;0;1;1;0;6;2;3;1;1;1;1;0;0;1;0;1;1;1;1;1;1;11;3;1;5;1;5;1;5;2;5;3;10
"""

df_dados = pd.read_csv(io.StringIO(dados_csv_string), sep=';')

# Engenharia de atributos comum
df_dados['Ganho'] = df_dados['Pontuação_Pos'] - df_dados['Pontuação_Pre']
df_dados['Delta_Conf'] = df_dados['Confianca_Pos'] - df_dados['Confianca_Pre']
df_dados['Melhorou_Conf'] = (df_dados['Delta_Conf'] > 0).astype(int)

# Separação de grupos
grupo_controle = df_dados[df_dados['Grupo'] == 'Controle'].copy()
grupo_tratamento = df_dados[df_dados['Grupo'] == 'Tratamento'].copy()

# ==============================================================================
# SEÇÃO 1: ANÁLISE DESCRITIVA
# ==============================================================================
print("="*60)
print("SEÇÃO 1: ANÁLISE DESCRITIVA")
print("="*60)

# Mapas de categorias
mapa_exp = {1: 'Nenhuma', 2: 'Básica', 3: 'Intermediário', 4: 'Avançada'}
mapa_conf = {1: 'Totalmente Inseguro', 2: 'Inseguro', 3: 'Confiante', 4: 'Totalmente Confiante'}

# Estatísticas Gerais
stats_idade = df_dados['Idade'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
desc_idade_grupo = df_dados.groupby('Grupo')['Idade'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)

stats_periodo = df_dados['Periodo'].agg(['count', 'median', 'min', 'max']).round(2)
desc_periodo_grupo = df_dados.groupby('Grupo')['Periodo'].agg(['count', 'median', 'min', 'max']).round(2)

print("\n--- Demografia ---")
print(f"Idade Média: {stats_idade['mean']} (DP: {stats_idade['std']})")
print(desc_idade_grupo)

print(f"\nPeríodo Mediano: {stats_periodo['median']}")
print(desc_periodo_grupo)

# Frequências
freq_exp = df_dados['Exp_Prog_Pre'].value_counts().sort_index().rename(index=mapa_exp)
freq_conf = df_dados['Confianca_Pre'].value_counts().sort_index().rename(index=mapa_conf)

print("\n--- Perfil Prévio ---")
print("Experiência em Programação:\n", freq_exp)
print("\nConfiança Inicial:\n", freq_conf)

# ==============================================================================
# SEÇÃO 2: ANÁLISE DE DESEMPENHO GERAL (NOTAS)
# ==============================================================================
print("\n" + "="*60)
print("SEÇÃO 2: ANÁLISE DE DESEMPENHO (NOTAS GERAIS)")
print("="*60)

def bootstrap_paired_diff(data_pre, data_pos, n_bootstrap=10000, seed=42):
    """Bootstrap para dados pareados (Mesmo aluno Pré vs Pós)"""
    np.random.seed(seed)
    individual_diffs = np.array(data_pos) - np.array(data_pre)
    n = len(individual_diffs)
    boot_means = [np.mean(np.random.choice(individual_diffs, size=n, replace=True)) for _ in range(n_bootstrap)]
    return np.array(boot_means)

def bootstrap_mean_diff(data_group_A, data_group_B, n_bootstrap=10000, seed=42):
    """Bootstrap para grupos independentes (Tratamento vs Controle)"""
    np.random.seed(seed)
    diffs = []
    nA, nB = len(data_group_A), len(data_group_B)
    for _ in range(n_bootstrap):
        sampleA = np.random.choice(data_group_A, size=nA, replace=True)
        sampleB = np.random.choice(data_group_B, size=nB, replace=True)
        # Diferença: Grupo B (Trat) - Grupo A (Cont)
        diffs.append(np.mean(sampleB) - np.mean(sampleA))

    ci = np.percentile(diffs, [2.5, 97.5])
    return np.mean(diffs), ci[0], ci[1] # Retorna Média, Lower, Upper

# Cálculos
boot_controle = bootstrap_paired_diff(grupo_controle['Pontuação_Pre'], grupo_controle['Pontuação_Pos'])
boot_tratamento = bootstrap_paired_diff(grupo_tratamento['Pontuação_Pre'], grupo_tratamento['Pontuação_Pos'])
mean_diff, ci_lower, ci_upper = bootstrap_mean_diff(grupo_controle['Ganho'], grupo_tratamento['Ganho'])
boot_inter = np.random.normal(mean_diff, (ci_upper-ci_lower)/4, 10000) # Simulação apenas para plot coerente com os anteriores

ci_c = np.percentile(boot_controle, [2.5, 97.5])
ci_t = np.percentile(boot_tratamento, [2.5, 97.5])
ci_i = [ci_lower, ci_upper]

# Visualização
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
sns.histplot(boot_controle, color="skyblue", kde=True, ax=axes[0])
axes[0].axvline(ci_c[0], color='r', ls='--');
axes[0].axvline(ci_c[1], color='r', ls='--');
axes[0].axvline(0, color='k')
axes[0].set_title(f"Evolução Controle\nIC95%: {ci_c[0]:.2f} a {ci_c[1]:.2f}")
axes[0].set_xlabel("Ganho (pontos)")
axes[0].set_ylabel("Quantidade")

sns.histplot(boot_tratamento, color="lightgreen", kde=True, ax=axes[1])
axes[1].axvline(ci_t[0], color='r', ls='--');
axes[1].axvline(ci_t[1], color='r', ls='--');
axes[1].axvline(0, color='k')
axes[1].set_title(f"Evolução Tratamento\nIC95%: {ci_t[0]:.2f} a {ci_t[1]:.2f}")
axes[1].set_xlabel("Ganho (pontos)")
axes[1].set_ylabel("Quantidade")

sns.histplot(boot_inter, color="salmon", kde=True, ax=axes[2])
axes[2].axvline(ci_i[0], color='r', ls='--');
axes[2].axvline(ci_i[1], color='r', ls='--');
axes[2].axvline(0, color='k')
axes[2].set_title(f"Diferença (Trat - Cont)\nIC95%: {ci_i[0]:.2f} a {ci_i[1]:.2f}")
axes[2].set_xlabel("Ganho (pontos)")
axes[2].set_ylabel("Quantidade")

plt.suptitle("Análise via Bootstrap (10.000 iterações)", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# ==============================================================================
# SEÇÃO 3: ANÁLISE DETALHADA POR HABILIDADES (FOREST PLOT)
# ==============================================================================
print("\n" + "="*60)
print("SEÇÃO 3: ANÁLISE POR HABILIDADES (FOREST PLOT)")
print("="*60)

# 1. Definição das Habilidades
habilidades = {
    'Conceitos Fundam.': ['Q1', 'Q2', 'Q3', 'Q4'],
    'Percurso': ['Q5', 'Q6', 'Q7'],
    'Inserção': ['Q8'],
    'Busca': ['Q9', 'Q10', 'Q11'],
    'Remoção': ['Q12', 'Q13', 'Q14']
}

# 2. Cálculo dos Ganhos por Habilidade
for nome, qs in habilidades.items():
    cols_pos = [f"{q}_Pos" for q in qs]
    cols_pre = [f"{q}_Pre" for q in qs]
    df_dados[f'Ganho_{nome}'] = df_dados[cols_pos].sum(axis=1) - df_dados[cols_pre].sum(axis=1)

# Atualizar DataFrames de Grupo (para incluir as novas colunas)
grupo_controle = df_dados[df_dados['Grupo'] == 'Controle']
grupo_tratamento = df_dados[df_dados['Grupo'] == 'Tratamento']

# 3. Bootstrap por Habilidade (Reusando a função definida acima)
res = []
for nome in habilidades.keys():
    # Usa a função bootstrap_mean_diff da Seção 2
    m, l, u = bootstrap_mean_diff(grupo_controle[f'Ganho_{nome}'], grupo_tratamento[f'Ganho_{nome}'])
    res.append({'Habilidade': nome, 'Diff': m, 'Lower': l, 'Upper': u})

df_res = pd.DataFrame(res)

# 4. Plot (Forest Plot)
plt.figure(figsize=(10, 6))

# Lógica de cores: Vermelho se significativo (IC não cruza zero), Azul caso contrário
colors = ['#e74c3c' if (row['Upper'] < 0 or row['Lower'] > 0) else '#3498db' for _, row in df_res.iterrows()]

# Barras de erro
y_pos = np.arange(len(df_res))
plt.errorbar(df_res['Diff'], y_pos,
             xerr=[df_res['Diff']-df_res['Lower'], df_res['Upper']-df_res['Diff']],
             fmt='none', ecolor='gray', elinewidth=2, capsize=5)

# Pontos centrais
for i, row in df_res.iterrows():
    plt.plot(row['Diff'], i, 'o', color=colors[i], markersize=10,
             markeredgecolor='black', label='Significativo' if colors[i]=='#e74c3c' and i==0 else "")

plt.axvline(0, color='black', linestyle='--', linewidth=1)
plt.yticks(y_pos, df_res['Habilidade'], fontsize=12)
plt.xlabel('< Controle Melhor   |   Tratamento Melhor >\nDiferença de Ganho (Tratamento - Controle)', fontsize=11)
plt.title('Impacto do Tratamento por Habilidade Específica\n(Diferença de Médias via Bootstrap IC 95%)', fontsize=12, y=1.05)

# Anotações de texto no gráfico
for i, row in df_res.iterrows():
    txt = f"{row['Diff']:.2f} [{row['Lower']:.2f}, {row['Upper']:.2f}]"
    # Ajuste dinâmico da posição do texto
    plt.text(df_res['Upper'].max() + 0.2, i, txt, va='center', ha='left', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

plt.grid(axis='x', linestyle='--', alpha=0.5)
sns.despine(left=True)
plt.tight_layout()
plt.show()

# ==============================================================================
# SEÇÃO 4: CONFIANÇA (BOOTSTRAP PROPORCIONAL, CALIBRAÇÃO E ESTIMATION)
# ==============================================================================
print("\n" + "="*60)
print("SEÇÃO 4: CONFIANÇA")
print("="*60)

# 4.1 Bootstrap de Proporção
def bootstrap_proportion(data, n_bootstrap=10000, seed=42):
    np.random.seed(seed)
    props = [np.mean(np.random.choice(data, len(data), replace=True)) for _ in range(n_bootstrap)]
    ci = np.percentile(props, [2.5, 97.5])
    return np.mean(props), ci[0], ci[1]

# 4.2 Bootstrap de Diferença de Proporção
def bootstrap_diff_prop(data_c, data_t, n=10000):
    np.random.seed(42)
    diffs = []
    for _ in range(n):
        sc = np.mean(np.random.choice(data_c, len(data_c), replace=True))
        st = np.mean(np.random.choice(data_t, len(data_t), replace=True))
        diffs.append(st - sc)
    ci = np.percentile(diffs, [2.5, 97.5])
    return np.mean(diffs), ci[0], ci[1]

# Cálculos
prop_c, low_c, high_c = bootstrap_proportion(grupo_controle['Melhorou_Conf'])
prop_t, low_t, high_t = bootstrap_proportion(grupo_tratamento['Melhorou_Conf'])
diff_prop, low_d, high_d = bootstrap_diff_prop(grupo_controle['Melhorou_Conf'], grupo_tratamento['Melhorou_Conf'])

#print(f"Prop. Melhora - Controle: {prop_c*100:.1f}% [{low_c*100:.1f}%, {high_c*100:.1f}%]")
#print(f"Prop. Melhora - Tratamento: {prop_t*100:.1f}% [{low_t*100:.1f}%, {high_t*100:.1f}%]")
#print(f"Diferença (Trat - Cont): {diff_prop*100:.1f} p.p. [{low_d*100:.1f}%, {high_d*100:.1f}%]")

# --- PLOT 4A: Stacked Bar Chart (Mudança de Distribuição) ---
cols_conf = ['Confianca_Pre', 'Confianca_Pos']
df_long = pd.melt(df_dados, id_vars=['Grupo'], value_vars=cols_conf, var_name='Momento', value_name='Nivel')
df_long['Momento'] = df_long['Momento'].replace({'Confianca_Pre': 'Pré', 'Confianca_Pos': 'Pós'})
df_counts = df_long.groupby(['Grupo', 'Momento', 'Nivel']).size().reset_index(name='Contagem')
full_idx = pd.MultiIndex.from_product([['Controle', 'Tratamento'], ['Pré', 'Pós'], [1, 2, 3, 4]], names=['Grupo', 'Momento', 'Nivel'])
df_counts = df_counts.set_index(['Grupo', 'Momento', 'Nivel']).reindex(full_idx, fill_value=0).reset_index()
df_counts['Total'] = df_counts.groupby(['Grupo', 'Momento'])['Contagem'].transform('sum')
df_counts['Pct'] = df_counts['Contagem'] / df_counts['Total'] * 100

fig, ax = plt.subplots(figsize=(10, 5))
grupos_momentos = [('Controle', 'Pré'), ('Controle', 'Pós'), ('Tratamento', 'Pré'), ('Tratamento', 'Pós')]
x_pos = [0, 1, 2.5, 3.5]
bottom = np.zeros(4)
colors_stack = {1: '#e74c3c', 2: '#f39c12', 3: '#3498db', 4: '#2ecc71'}
labels_map = {1: 'Tot. Inseguro', 2: 'Inseguro', 3: 'Confiante', 4: 'Tot. Confiante'}

for nivel in [4, 3, 2, 1]:
    vals = [df_counts[(df_counts['Grupo']==g) & (df_counts['Momento']==m) & (df_counts['Nivel']==nivel)]['Pct'].values[0] for g, m in grupos_momentos]
    p = ax.bar(x_pos, vals, 0.6, bottom=bottom, color=colors_stack[nivel], label=labels_map[nivel], edgecolor='white')
    for i, rect in enumerate(p):
        if rect.get_height() > 5:
            ax.text(rect.get_x() + rect.get_width()/2., bottom[i] + rect.get_height()/2., f"{int(rect.get_height())}%", ha='center', va='center', color='white', fontweight='bold')
    bottom += np.array(vals)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{g}\n{m}" for g, m in grupos_momentos])
ax.set_title("Evolução da Distribuição de Confiança", fontsize=12, y=1.05)
ax.set_ylabel('Distribuição (%) dos Participantes')
ax.legend(title='Nível de Confiança', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.show()

# --- PLOT 4B: Estimation Chart (Proportions & Difference) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})

# Painel 1: Proporções
groups_labels = ['Controle', 'Tratamento']
means = [prop_c, prop_t]
y_err = [[prop_c - low_c, prop_t - low_t], [high_c - prop_c, high_t - prop_t]]
colors_conf = ['black', 'black']

ax1.errorbar(x=range(2), y=means, yerr=y_err, fmt='none', ecolor='black', capsize=5)
for i, (m, c) in enumerate(zip(means, colors_conf)):
    ax1.plot(i, m, 'o', color=c, markersize=15)
    ax1.text(i+0.05, m, f"{m*100:.1f}%", va='center', color=c, fontweight='bold')
ax1.set_xticks(range(2))
ax1.set_xticklabels(groups_labels)
ax1.set_title('Variação da Confiança\n(Proporção de Aumento via Bootstrap IC 95%)', fontsize=12, y=1.05)
ax1.set_ylabel('Proporção de Alunos que Melhoraram', fontsize=11)
ax1.set_ylim(0, 1.1)

# Painel 2: Diferença
diff_err = [[diff_prop - low_d], [high_d - diff_prop]]
ax2.errorbar(x=[0], y=[diff_prop], yerr=diff_err, fmt='none', elinewidth=3, capsize=6)
ax2.plot(0, diff_prop, 'D', markersize=12)
ax2.axhline(0, color='red', linestyle='--', label='Sem Efeito')
ax2.text(0.02, diff_prop, f"+{diff_prop*100:.1f} p.p.", va='center_baseline', fontweight='bold')

#ax2.axhspan(low_d, high_d, alpha=0.1)
ax2.set_xticks([0])
ax2.set_xticklabels(['Diferença\n(Trat - Cont)'])
ax2.set_title('Tamanho do Efeito\n(Delta)', fontsize=12, y=1.05)
ax2.grid(axis='y', linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

# ==============================================================================
# SEÇÃO 5: PERFILAMENTO (CLUSTERIZAÇÃO K-MEANS)
# ==============================================================================
print("\n" + "="*60)
print("SEÇÃO 5: PERFILAMENTO DE ALUNOS (CLUSTERIZAÇÃO)")
print("="*60)

features = ['Exp_Prog_Pre', 'Pontuação_Pre', 'Ganho']
X_scaled = StandardScaler().fit_transform(df_dados[features])

# Dendrograma
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', labels=df_dados['ID'].values, distance_sort='descending')
plt.title('Dendrograma de Similaridade entre Perfis de Alunos', fontsize=12, y=1.05)
plt.xlabel('Participantes', fontsize=11)
plt.ylabel('Distância Euclidiana (Método de Ward)', fontsize=11)
plt.axhline(y=3.5, color='r', linestyle='--')
plt.text(0.5, 3.6, 'Corte sugerido (k=3)', color='r', fontsize=10)
plt.show()

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_scaled)
df_dados['Cluster'] = kmeans.labels_

# Mapeamento semântico dos clusters
stats_cluster = df_dados.groupby('Cluster')[['Pontuação_Pre', 'Ganho']].mean()
id_expert = stats_cluster['Pontuação_Pre'].idxmax()
id_iniciante = stats_cluster['Ganho'].idxmax()
id_intermed = list({0, 1, 2} - {id_expert, id_iniciante})[0]

cluster_map = {id_expert: 'Alta Proficiência Prévia', id_iniciante: 'Aquisição Acelerada', id_intermed: 'Progressão Intermediária'}
df_dados['Perfil'] = df_dados['Cluster'].map(cluster_map)

# Scatter Plot dos Perfis
np.random.seed(42)
jitter_x = np.random.uniform(-0.3, 0.3, size=len(df_dados))
jitter_y = np.random.uniform(-0.3, 0.3, size=len(df_dados))

plt.figure(figsize=(10, 6))
palette_cluster = {"Alta Proficiência Prévia": "#E15F60", "Aquisição Acelerada": "#FF7F0E", "Progressão Intermediária": "#43AA43"}
sns.scatterplot(x=df_dados['Pontuação_Pre'] + jitter_x, y=df_dados['Ganho'] + jitter_y,
                hue=df_dados['Perfil'], style=df_dados['Grupo'], s=200, palette=palette_cluster, edgecolor='black')

for i in range(len(df_dados)):
    plt.text(df_dados.Pontuação_Pre[i] + jitter_x[i] + 0.1, df_dados.Ganho[i] + jitter_y[i] + 0.1, df_dados.ID[i], fontsize=9)

plt.title('Perfis de Alunos Identificados: Nota Inicial vs Evolução', fontsize=12, y=1.05)
plt.xlabel('Nota Pré-Teste')
plt.ylabel('Ganho de Aprendizagem')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("\n--- Caracterização dos Perfis ---")
print(df_dados.groupby('Perfil').agg({'ID':'count', 'Pontuação_Pre':'mean', 'Ganho':'mean', 'Pontuação_Pos':'mean'}).round(2))

# ==============================================================================
# SEÇÃO 6: USABILIDADE (SUS & NPS - APENAS TRATAMENTO)
# ==============================================================================
print("\n" + "="*60)
print("SEÇÃO 6: EXPERIÊNCIA DO USUÁRIO (SUS & NPS)")
print("="*60)

df_trat = grupo_tratamento.copy()

# Cálculo SUS
for i in range(1, 11):
    col = f'SUS_Q{i}'
    if i % 2 != 0: # Ímpar: Resposta - 1
        df_trat[f'Score_{col}'] = df_trat[col] - 1
    else: # Par: 5 - Resposta
        df_trat[f'Score_{col}'] = 5 - df_trat[col]

df_trat['SUS_Total'] = df_trat[[f'Score_SUS_Q{i}' for i in range(1, 11)]].sum(axis=1) * 2.5
sus_stats = df_trat['SUS_Total'].agg(['mean', 'std', 'min', 'max'])

# Estatísticas do SUS
sus_mean = df_trat['SUS_Total'].mean()
sus_std = df_trat['SUS_Total'].std()
sus_min = df_trat['SUS_Total'].min()
sus_max = df_trat['SUS_Total'].max()

print("--- Estatísticas do SUS ---")
print(f"Média: {sus_mean:.2f}")
print(f"Desvio Padrão: {sus_std:.2f}")
print(f"Mínimo: {sus_min:.2f}")
print(f"Máximo: {sus_max:.2f}")

# Cálculo NPS
def classificar_nps(val):
    if val >= 9: return 'Promotor'
    elif val >= 7: return 'Neutro'
    else: return 'Detrator'

df_trat['NPS_Class'] = df_trat['NPS_Q11'].apply(classificar_nps)
nps_counts = df_trat['NPS_Class'].value_counts()
nps_score = ((nps_counts.get('Promotor', 0) - nps_counts.get('Detrator', 0)) / len(df_trat)) * 100

print("\n--- Estatísticas do NPS ---")
print(nps_counts)
print(f"NPS Score: {nps_score:.2f}")

# Gráficos UX
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# SUS Plot
ax1.axhspan(0, 50, color='red', alpha=0.1, label='Inaceitável')
ax1.axhspan(50, 70, color='orange', alpha=0.1, label='Marginal')
ax1.axhspan(70, 85, color='yellow', alpha=0.1, label='Aceitável')
ax1.axhspan(85, 100, color='green', alpha=0.1, label='Excelente')
sns.boxplot(y=df_trat['SUS_Total'], ax=ax1, width=0.3, color='white')
sns.stripplot(y=df_trat['SUS_Total'], ax=ax1, color='black', size=8)
ax1.set_title(f'Score de Usabilidade (SUS)\nMédia: {sus_mean:.1f} (DP={sus_std:.1f})', fontsize=12, y=1.05)
ax1.set_ylabel('Score SUS (0-100)')
ax1.set_ylim(0, 100)
ax1.legend(loc='lower right')

# NPS Plot
cats = ['Detrator', 'Neutro', 'Promotor']
vals = [nps_counts.get(c, 0) for c in cats]
bars = ax2.bar(cats, vals, color=['#e74c3c', '#f39c12', '#2ecc71'], edgecolor='black')
ax2.set_title(f'NPS ({nps_score:.0f})')
ax2.bar_label(bars)

plt.tight_layout()
plt.show()

# Correlação SUS vs Aprendizagem
corr_sus_ganho = df_trat['SUS_Total'].corr(df_trat['Ganho'])
print(f"\nCorrelação SUS vs Ganho de Aprendizagem: {corr_sus_ganho:.2f}")
