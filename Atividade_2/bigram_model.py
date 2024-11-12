import nltk
import torch
import re


class Bigram:
    def __init__(self):
        self.b = {}
        self.stoi = {}

    def build_tokens_matrix(self, train_data):
        fdist1 = nltk.FreqDist(train_data)
        self.word_unique_count = fdist1.B()
        print(self.word_unique_count)

        self.end_position = list(fdist1.keys()).index("<end>")

        self.N = torch.zeros(
            (self.word_unique_count, self.word_unique_count), dtype=torch.int32
        )

        self.stoi = {s: i for i, s in enumerate(fdist1.keys())}
        self.itos = {i: s for s, i in self.stoi.items()}

    def train(self, train_data):
        self.build_tokens_matrix(train_data)

        for sent1, sent2 in zip(train_data, train_data[1:]):
            ix1 = self.stoi[sent1]
            ix2 = self.stoi[sent2]

            self.N[ix1, ix2] += 1

        self.P = self.N.float()
        self.P = self.P / self.P.sum(1, keepdim=True)

    def generator(self, n_samples):
        g = torch.Generator().manual_seed(42)

        for _ in range(n_samples):
            out = []
            ix = 0

            while True:
                p = self.P[ix]
                ix = torch.multinomial(
                    p, num_samples=1, replacement=True, generator=g
                ).item()

                if not bool(
                    re.search(r"[!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~]", self.itos[ix])
                ):
                    out.append(" ")

                out.append(self.itos[ix])

                if ix == self.end_position:
                    break

            print("".join(out))

    def calculate_perplexity(self, test_data):
        """
        Calcula a perplexidade de um texto de teste com base nas probabilidades de bigramas.

        Parâmetros:
        - bigram_probs: dicionário contendo as probabilidades dos bigramas, no formato {(w1, w2): probabilidade}.
        - test_text: lista de tokens (palavras) do texto de teste.

        Retorno:
        - perplexidade (float)
        """
        N = len(test_data) - 1
        prob_total = 1.0

        for sent1, sent2 in zip(test_data, test_data[1:]):
            ix1 = self.stoi.get(sent1, None)
            ix2 = self.stoi.get(sent2, None)

            if ix1 == None or ix2 == None:
                bigram_prob = 1e-10
            else:
                bigram_prob = self.P[ix1, ix2]

            prob_total *= bigram_prob

        perplexity = prob_total ** (-1 / N)

        print(perplexity)
        return perplexity

    def save_bigram_probabilities(self):
        # torch.save(self.P, "bigrams_probs.pt")
        with open("bigrams_probs.txt", "w", encoding="utf-8") as f:
            for i, row in enumerate(self.P):
                for j, val in enumerate(row):
                    f.write(f"({self.itos[i]}, {self.itos[j]}: {val} ) ")
                f.write("\n")
