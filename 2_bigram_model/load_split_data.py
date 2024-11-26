# Diretório do corpo de texto a ser utilizado, presente no mesmo diretório

import json
import os
import random

import nltk

from tqdm import tqdm


class CorpusLoader:
    """
    Classe responsável por ler os documentos de texto e realizar a separação
    entre dados de treino e teste

    Parâmetros:
    - type: tipo de tokenização da classe, se será word ou sentence.
    """

    def __init__(self, type):
        self.type = type
        self.files_processed = []
        self.total_token_count = 0
        self.data = []

        if self.type == "word":
            print("Divisão de tokens por palavras")
        else:
            print("Divisão de tokens por sentenças")

    def read_files_chunk(self, files_path):
        """
        Realiza a leitura individual de cada arquivo e realiza a tokenização

        Parâmetros:
        - files_path: recebe lista de nomes dos arquivos a serem lidos.

        Retornos:
        - files: lista de arquivos lidos
        - token_count: quantia de tokens lidos
        """
        files = []
        token_count = 0

        for file_path in tqdm(files_path):
            with open(file_path, "r", encoding="utf-8") as file:
                # print(f"Lendo arquivo: {file_path}")
                file_data = json.load(file)

                # Verifica se documento há texto
                if file_data["text"]:

                    # Realiza a verificação do tipo escolhido para realizar
                    # a tokenização apropriada
                    if self.type == "word":
                        file_data["text_tokens"] = nltk.tokenize.word_tokenize(
                            file_data["text"], language="portuguese"
                        )
                    else:
                        file_data["text_tokens"] = nltk.tokenize.sent_tokenize(
                            file_data["text"], language="portuguese"
                        )

                    # Como na biblioteca NLTK não há os tokens de início/fim
                    # estes são adicionados ao início e fim dos textos
                    file_data["text_tokens"].insert(0, "<start>")
                    file_data["text_tokens"].append("<end>")

                    # Contabiliza a quantia de tokens por documento
                    file_data["token_count"] = len(file_data["text_tokens"])

                    # Somatório total da quantia de tokens dos documentos lidos
                    token_count += file_data["token_count"]
                else:
                    file_data["text_tokens"] = []
                    file_data["token_count"] = 0

                files.append(file_data)

        return files, token_count

    def read_all_files(self, corpus_directory, files_quantity):
        """
        Leitura de nomes de arquivos presentes no diretório informado

        Parâmetros:
        - corpus_directory: Diretório do corpo de texto que será utilizado
        - files_quantity: quantia de arquivos presente no diretório que será lida

        Retornos:
        - files_processes: arquivos lidos e com devidos valores calculados
        """
        files_names = [
            os.path.join(corpus_directory, file)
            for file in os.listdir(corpus_directory)
            if file.endswith(".json")
        ]

        # Verifica se há quantidade delimitante
        if files_quantity != None:
            files_names = files_names[:files_quantity]

        # Lê o bloco de arquivos a partir da lista de nomes
        print(f"Lendo {len(files_names)} arquivos.")
        files, token_count = self.read_files_chunk(files_names)
        self.files_processed.extend(files)
        self.total_token_count += token_count

        return self.files_processed

    def split_test_train(self, split_proportion):
        """
        Separação de dados de treino e teste

        Parâmetros:
        - split_proportion: proporção de dados de treino a ser utilizada

        Retornos:
        - train_data: Lista de tokens de treino
        - test_data: Lista de tokens de teste
        """
        proportional_token_count = split_proportion * self.total_token_count

        train_data = []
        test_data = []

        count_tokens = 0

        # Embaralha os arquivos lidos em uma ordem aleatória
        shuffled_files_processed = self.files_processed
        random.seed(4)
        random.shuffle(shuffled_files_processed)

        print("Separando pedaços de teste e treino, aguarde...")
        for file in tqdm(shuffled_files_processed):

            # Como cada arquivo pode ter uma quantia de tokens diferente
            # para obter uma melhor porcentagem de separação vamos separar a porcentagem
            # escolhida por tokens e não arquivos

            # Calculamos a quantidade de tokens até o limiar desejado
            if count_tokens + file["token_count"] <= proportional_token_count:
                train_data += file["text_tokens"]
                count_tokens += file["token_count"]
            else:
                # O restante colocamos em dados de treino
                test_data += file["text_tokens"]

        print(f"Tamanho de texto de teste: {len(train_data)}")
        print(f"Tamanho de texto de treino: {len(test_data)}")

        return train_data, test_data
