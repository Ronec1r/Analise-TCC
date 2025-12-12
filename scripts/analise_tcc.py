import numpy as np

# 1. Dados das Notas (Pós-teste)
gc_pos = np.array([12, 13, 8, 10, 13])
ge_pos = np.array([10, 9, 10, 12, 8, 11])

# 2. Função de Bootstrap para Diferença de Médias
def bootstrap_diff(data1, data2, n_iterations=5000):
    diffs = []
    for _ in range(n_iterations):
        sample1 = np.random.choice(data1, size=len(data1), replace=True)
        sample2 = np.random.choice(data2, size=len(data2), replace=True)
        diffs.append(np.mean(sample1) - np.mean(sample2))
    return np.percentile(diffs, [2.5, 97.5])

ic_95 = bootstrap_diff(gc_pos, ge_pos)
print(f"IC 95% para a diferença das médias (GC - GE): {ic_95}")

# 3. Cálculo do SUS (System Usability Scale)
# Respostas dos 6 alunos (GE)
respostas_sus = [
    [4,3,3,2,4,2,4,3,4,3], [5,2,4,3,3,1,4,1,5,5], [3,3,2,5,3,2,3,2,3,3],
    [4,1,4,2,5,1,5,1,5,1], [5,1,4,3,5,2,5,1,5,5], [5,2,5,4,5,2,5,1,5,1]
]

def calcular_sus(respostas):
    scores_finais = []
    for r in respostas:
        # Itens ímpares: r - 1 | Itens pares: 5 - r
        pontos = [(r[i]-1) if (i % 2 == 0) else (5-r[i]) for i in range(10)]
        scores_finais.append(sum(pontos) * 2.5)
    return np.mean(scores_finais)

print(f"Média Final SUS: {calcular_sus(respostas_sus)}")

# 4. Cálculo do NPS (Net Promoter Score)
nps_notas = [8, 8, 7, 10, 10, 10]
promotores = len([n for n in nps_notas if n >= 9])
detratores = len([n for n in nps_notas if n <= 6])
nps_score = ((promotores - detratores) / len(nps_notas)) * 100
print(f"NPS Final: {nps_score}")
