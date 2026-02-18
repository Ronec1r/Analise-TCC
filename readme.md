# Análise Estatística - TCC: Ensino de Estruturas de Dados com VisuAlgo

Este repositório contém o script de análise de dados utilizado no Trabalho de Conclusão de Curso (TCC) do Bacharelado em Ciência da Computação do Instituto Federal de Sergipe (IFS).

**Título do Trabalho:** Dificuldades no Ensino de Estruturas de Dados com ênfase em Árvores Binárias de Busca: Uma Análise Comparativa entre o Método Tradicional e o Uso de um Software Educativo.<br>
**Autor:** Rone Clay Oliveira Andrade<br>
**Ano:** 2025

## 📊 Sobre a Análise

O estudo utilizou uma abordagem quantitativa quase-experimental ($n=11$) comparando um Grupo Controle (Aula Expositiva) e um Grupo Experimental (Software VisuAlgo). Devido ao tamanho da amostra, foram aplicadas técnicas estatísticas computacionais para validação robusta dos resultados.

O script Python realiza:
1.  **Bootstrap (10.000 iterações):** Para gerar Intervalos de Confiança (IC 95%) das médias de ganho de aprendizagem, contornando a limitação da normalidade em amostras pequenas.
2.  **Forest Plot (Tamanho de Efeito):** Visualização da diferença de desempenho segmentada por competência cognitiva (Fundamentos, Recursão, Operações).
3.  **Análise de Usabilidade (UX):** Processamento dos dados da escala SUS (*System Usability Scale*) e NPS (*Net Promoter Score*).
4.  **Clusterização (K-Means):** Agrupamento não-supervisionado dos alunos para identificar perfis de aprendizagem baseados em conhecimento prévio vs. ganho.

## 🚀 Como Executar

O código foi projetado para rodar em ambientes Notebook (Jupyter ou Google Colab) ou como script Python standalone.

### Pré-requisitos

As seguintes bibliotecas Python são necessárias:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
