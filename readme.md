  # Análise da Ferramenta VisuAlgo no Ensino de Árvores Binárias de Busca

Este repositório contém o conjunto de dados anonimizados e o script de análise estatística referente ao Trabalho de Conclusão de Curso (TCC) desenvolvido no **Instituto Federal de Sergipe (IFS) - Campus Itabaiana**.

O objetivo da pesquisa foi analisar a influência da ferramenta de visualização *VisuAlgo* no desempenho académico e na satisfação de estudantes de licenciatura durante a aprendizagem de Árvores Binárias de Busca (BST).

## 📂 Estrutura do Repositório



* `data/`: Contém o ficheiro `dados_anonimizados.csv` com as notas e respostas dos questionários.
* `scripts/`: Contém o script `analise_tcc.py` utilizado para o processamento dos dados.
* `README.md`: Instruções e documentação do projeto.

## 📊 Dicionário de Dados

O ficheiro `data/dados_anonimizados.csv` utiliza a seguinte estrutura:

| Variável | Descrição |
| :--- | :--- |
| `id_aluno` | Identificador numérico único para garantir o anonimato dos participantes. |
| `grupo` | Classificação do participante: `controle` (aula tradicional) ou `experimental` (com VisuAlgo). |
| `pre_teste` | Nota obtida na avaliação diagnóstica inicial. |
| `pos_teste` | Nota obtida na avaliação após a intervenção pedagógica. |
| `nps` | Pontuação de 0 a 10 para o cálculo do *Net Promoter Score* (apenas grupo experimental). |
| `sus_q1` a `sus_q10` | Respostas (1-5) para as 10 perguntas da *System Usability Scale* (apenas grupo experimental). |

## ⚙️ Como Reproduzir a Análise

Para executar os cálculos estatísticos (Bootstrap, SUS e NPS), precisará do Python instalado no seu ambiente.

1.  **Instale as dependências necessárias:**
    ```bash
    pip install numpy pandas
    ```

2.  **Execute o script de análise:**
    ```bash
    python scripts/analise_estatistica.py
    ```

## 🧪 Metodologia Estatística

Devido ao tamanho reduzido da amostra ($n=11$), a análise inferencial foi realizada utilizando a técnica de **Bootstrap** com 5.000 reamostragens para o cálculo dos Intervalos de Confiança (IC 95%). Esta técnica permite maior robustez científica em amostras pequenas. 

A usabilidade foi medida através do protocolo SUS de Brooke (1996) e a satisfação via NPS (*Net Promoter Score*).

## ⚖️ Ciência Aberta e Ética

Este projeto segue os princípios da **Ciência Aberta** (*Open Science*), disponibilizando dados e algoritmos para garantir a reprodutibilidade da pesquisa. 

Em conformidade com as diretrizes éticas, todos os dados foram anonimizados. Nomes, e-mails ou quaisquer
