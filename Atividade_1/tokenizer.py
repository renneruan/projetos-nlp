from helper_functions import count_pairs, merge_tokens


class BPETokenizer:
    """
    Classe com funções de um tokenizador aplicando algoritmo de BPE

    A partir de um corpo de teste, treina o tokenizador criando um vocabulário de tokens.

    O vocabulário irá variar de acordo com o parâmetro de número de iterações/merge.

    Este tokenizador tem como padrão um vocabulário de um 256 tokens, uma vez que um
        caractere em UTF-8 utiliza um byte que pode possuir 256 valores.

    Attributes:
        num_merges (int): Quantia de merges/junções de token a serem realizadas.

    Example:
        bpe = BPETokenizer(20)
        bpe.train("texto")
    """

    def __init__(self, num_merges: int):
        # Inicialização dos valores do tokenizador
        # 256 valores possíveis em um byte de acordo com UTF-8
        # Textos complexos utilizam mais de um byte para representação
        self.byte_value_offset = 256

        self.num_merges = num_merges
        self.vocab_size = num_merges + self.byte_value_offset

        self.tokens = []
        self.processed_tokens = []

        # Dicionário de merges a ser utilizado para encode
        self.merges = {}
        # Vocabulário a ser utilizado em decode
        self.vocabulary = {}

    def train(self, text):
        """
        Realiza o treinamento do tokenizador.
        A partir do corpo de texto recebido constrói o novo vocabulário de tokens.
        Os tokens fundidos terão sua referência salva em dicionário de junções.

        A quantia de iterações a ser realizadas para construção do vocabulário
          será a de declaração do Tokenizador.

        Args:
            text (str): Texto que será utilizado para criação de vocabulário.
        """

        print(f"Tamanho do texto recebido em caracteres: {len(text)}")

        # Transforma o texto recebido em bytes
        self.tokens = text.encode("utf-8")
        # Transforma bytes para inteiros entre 0 e 255
        self.tokens = list(map(int, self.tokens))

        print(f"Tamanho do texto recebido em bytes: {len(self.tokens)}")

        self.processed_tokens = self.tokens

        for i in range(self.num_merges):
            counts = count_pairs(self.processed_tokens)

            # Resgata o valor de tupla adjacente com maior quantia de ocorrências
            top_pair = max(counts, key=counts.get)

            new_token_value = self.byte_value_offset + i

            # print(f"Junção de {top_pair} em {new_token_value}")

            # Junta os valores adjacentes mais frequentes em um novo valor de token
            self.processed_tokens = merge_tokens(
                self.processed_tokens, top_pair, new_token_value
            )

            # Salva a referência de junção, dos valores antigos para o novo
            self.merges[top_pair] = new_token_value

        print(f"Tamanho da lista de tokens após BPE: {len(self.processed_tokens)}")

        taxa = len(self.tokens) / len(self.processed_tokens)
        print(f"Taxa de compressão (Tokens originais/Tokens BPE): {taxa:.2f}X")

        self.vocabulary = {idx: bytes([idx]) for idx in range(self.byte_value_offset)}
        for (p0, p1), idx in self.merges.items():
            self.vocabulary[idx] = self.vocabulary[p0] + self.vocabulary[p1]

    def decode(self, input_tokens):
        """
        Transforma uma lista de tokens recebido em string.

        Utiliza o mapeamento previamente salvo após treinamento do tokenizador.

        Args:
            input_tokens (list): Lista de tokens a serem decodificados.
        Returns:
            text (str): Texto que representa a lista de tokens recebida.
        """
        tokens = b"".join(self.vocabulary[idx] for idx in input_tokens)
        text = tokens.decode("utf-8", errors="replace")

        return text

    def encode(self, text):
        """
        Transforma um texto em uma lista de tokens.

        Utiliza o vocabulário de tokens salvo após treinamento do tokenizador.

        Args:
            text (str): Texto a ser codificado.
        Returns:
            tokens (list): Lista de tokens que representam o texto.
        """
        tokens = list(text.encode("utf-8"))
        while True:
            stats = count_pairs(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break

            new_token = self.merges[pair]
            tokens = merge_tokens(tokens, pair, new_token)

        return tokens

    def save(self, file_prefix, only_new_tokens=True, verbose=True):
        """
        Salva os tokens gerados em um arquivo txt para melhor visualização.

        Args:
            file_prefix (str): Nome do arquivo txt a ser gerado
            only_new_tokens (boolean): Opção para salvar apenas novos tokens gerados pelo BPE
                Caso seja False irá salvar no txt todos os tokens do vocabulário incluindo os
                256 tokens padrões para UTF-8.
            verbose (boolean): Se verdadeiro mostra o resultado dos tokens salvos em arquivo.
        """
        file = file_prefix + "_" + self.num_merges + "tokens.txt"

        # Transforma a tupla de valores de merge para um dicionário.
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}

        vocabulary_list = self.vocabulary.items()
        with open(file, "w", encoding="utf-8") as f:
            for idx, token in vocabulary_list:
                s = token.decode("utf-8", errors="replace")

                output = False
                if idx in inverted_merges:
                    # Salva em arquivo no caso de token com junções
                    # Resgata a partir do dicionário de merges os tokens filhos.
                    idx0, idx1 = inverted_merges[idx]

                    s0 = self.vocabulary[idx0].decode("utf-8", errors="replace")
                    s1 = self.vocabulary[idx1].decode("utf-8", errors="replace")

                    output = f"{idx}: [{s0}][{s1}] -> [{s}]"
                elif not only_new_tokens:
                    # Salva em arquivo no caso de caracter único (256 iniciais)
                    output = f"[{s}] {idx}"

                if output:
                    if verbose:
                        print(output)
                    f.write(output + "\n")
