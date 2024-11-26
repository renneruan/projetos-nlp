import nltk
import torch
import re
from tqdm import tqdm
import math


class Bigram:
    def __init__(self, smoothed=False):
        """
        Classe contendo modelo Bigrama com funções de treinamento, geração e cálculo de perplexidade.

        Parâmetros:
        - smoothed: Parâmetro para informar suavização das probabilidades do modelo.
        """

        self.smoothed = smoothed
        self.b = {}
        self.stoi = {}

    def build_tokens_matrix(self, train_data):
        """
        Cria a matriz de frequências com as funções do PyTorch
        Cria uma matriz com tamanho de acordo com a quantia de tokens únicos
        Esta função necessita de uma grande alocação de memória o que para a forma
        que os tokens únicos foram resgatados, limita a quantia de documentos utilizados

        Parâmetros:
        - train_data: lista de tokens (palavras) do texto de treino.
        """

        # Verifica quantia de tokens únicos
        fdist1 = nltk.FreqDist(train_data)
        self.word_unique_count = fdist1.B()
        print(f"Tokens únicos analisados: {self.word_unique_count}")

        # Resgata posição do token de fim de texto, usado posteriormente
        self.end_position = list(fdist1.keys()).index("<end>")

        # Cria uma matriz de zeros com as dimensões de acordo com o número de tokens únicos
        self.N = torch.zeros(
            (self.word_unique_count, self.word_unique_count), dtype=torch.int16
        )

        # Cria dicionários de correspondências, para traduzir bigramas para indexes e indexes para bigramas
        self.stoi = {s: i for i, s in enumerate(fdist1.keys())}
        self.itos = {i: s for s, i in self.stoi.items()}

    def clean_train_data(self, train_data):
        list_of_tokens = [
            x
            for x in train_data
            if (
                not (
                    bool(
                        re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", x)
                        and len(x) > 2
                    )
                )
            )
        ]
        return list_of_tokens  # Output: [1, 2, 3]

    def train(self, train_data):
        """
        Realiza a chamada para a criação de matriz de frequências
        Adiciona as frequência de cada bigrama na matriz criada

        Parâmetros:
        - train_data: lista de tokens (palavras) do texto de treino.
        """
        self.build_tokens_matrix(train_data)

        print("Realizando treinamento de modelo bigrama")
        for token1, token2 in tqdm(zip(train_data, train_data[1:])):
            ix1 = self.stoi[token1]
            ix2 = self.stoi[token2]

            self.N[ix1, ix2] += 1

        # O processamento das probabilidades era feito nesta função
        # resultando em uma matriz não de frequências, mas de probabilidades
        # porém isso necessitaria uma quantia de memória o suficiente para as
        # 2 matrizes de quantia_tokens X quantia_tokens criadas.

        # self.P = self.N.float()

        # smooth_constant = 0
        # if self.smoothed:
        #     smooth_constant = 1

        # self.P = (self.P + smooth_constant) / self.P.sum(1, keepdim=True)

    def generator(self, n_samples):
        """
        Gerador de texto a partir da matriz de frequências previamente treinada
        Calcula iterativamente a probabilidade do próximo token dado o token anterior

        Sempre irá começar a geração a partir do token 0 = <start>

        Parâmetros:
        - n_samples: Quantia de textos a serem gerados.
        """
        g = torch.Generator().manual_seed(42)

        # Verifica se modelo possui suavização, isso resulta em uma geração mais aleatória
        # tendo em vista que temos probabilidades baixas para o corpo de teste utilizado
        smooth_constant = 0
        if self.smoothed:
            smooth_constant = 1

        for _ in range(n_samples):
            out = []
            ix = 0

            while True:
                # print(p)
                # p = self.P[ix]

                # Calcula a probabilidade a partir da matriz de frequências
                # Calcula a probabilidade da linha de acordo com o bigrama inicial
                p = (self.N[ix] + smooth_constant).float() / self.N[ix].sum()

                # Resgata (sorteia) a partir da função multinomial o próximo bigrama
                ix = torch.multinomial(
                    p, num_samples=1, replacement=True, generator=g
                ).item()

                # Verifica se não é pontuação para concatenar espaços no texto gerado
                if not bool(
                    re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", self.itos[ix])
                ):
                    out.append(" ")

                out.append(self.itos[ix])

                # Se gerar o token de final de texto finaliza a geração
                if ix == self.end_position:
                    break

            print("".join(out))

    def calculate_perplexity(self, test_data):
        """
        Calcula a perplexidade de um texto de teste com base nas frequências de bigramas
        previamente calculadas na criação do modelo.

        Parâmetros:
        - test_data: lista de tokens (palavras) do texto de teste.

        Retorno:
        - perplexity: Valor da perplexidade (float)
        """

        # Quantia de tokens a serem analizados, reduzido de 1 por ser o token inicial
        N = len(test_data) - 1
        log_prob_total = 0.0

        # Verifica se há suavização no modelo
        smooth_constant = 0
        if self.smoothed:
            smooth_constant = 1

        sum_array = {}
        for token1, token2 in tqdm(zip(test_data, test_data[1:])):
            ix1 = self.stoi.get(token1, None)
            ix2 = self.stoi.get(token2, None)

            # Verifica se há ambos os tokens no dicionário de índices
            # Ou seja, se já apareceu nos dados de treino, se não
            # aplica um valor ínfimo para a probabilidade
            if ix1 == None or ix2 == None:
                bigram_prob = 1e-10
            else:
                # Verifica se já foi realizada a soma daquela linha visando otimização
                if ix1 in sum_array:
                    sum_row = sum_array[ix1]
                else:
                    sum_array[ix1] = self.N[ix1].sum()
                    sum_row = sum_array[ix1]
                p = (self.N[ix1].float() + smooth_constant) / sum_row

                # Somatório das probabilidades dos tokens de teste
                bigram_prob = p[ix2]

            # realiza o logarítimo do somatório uma vez que estamos trabalhando
            # com valores muito baixos
            log_prob_total += math.log(max(bigram_prob, 1e-10))

        # Como lob_prob_total terá a entropia, ao aplicarmos a exponenciação
        # obtemos a perplexidade
        perplexity = math.exp(-log_prob_total / N)

        return perplexity

    def save_bigram_probabilities(self):
        """
        Função para salvar a matriz criada para melhor visualização
        """

        # torch.save(self.P, "bigrams_probs.pt")
        with open("bigrams_probs.txt", "w", encoding="utf-8") as f:
            smooth_constant = 0
            if self.smoothed:
                smooth_constant = 1
            for i, row in enumerate(self.N):
                p = (row.float() + smooth_constant) / self.N.sum()

                for j, val in enumerate(p):
                    f.write(f"({self.itos[i]}, {self.itos[j]}: {val} ) ")
                f.write("\n")
