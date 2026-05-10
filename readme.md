# Análise Estatística: Ensino de Estruturas de Dados com VisuAlgo

Este repositório contém o script de análise de dados utilizado na pesquisa realizada.

**Título do Trabalho:** Impacto da Visualização Algorítmica no Aprendizado de Estruturas de Dados: Uma Análise Empírica com o VisuAlgo.<br>
**Ano:** 2026

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
