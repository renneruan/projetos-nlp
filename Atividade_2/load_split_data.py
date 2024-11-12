# Diretório do corpo de texto a ser utilizado, presente no mesmo diretório

import json
import os
import random

import nltk

from tqdm import tqdm


class CorpusLoader:
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
        files = []
        token_count = 0

        for file_path in tqdm(files_path):
            with open(file_path, "r", encoding="utf-8") as file:
                # print(f"Lendo arquivo: {file_path}")
                file_data = json.load(file)

                if file_data["text"]:
                    if self.type == "word":
                        file_data["text_tokens"] = nltk.tokenize.word_tokenize(
                            file_data["text"], language="portuguese"
                        )
                    else:
                        file_data["text_tokens"] = nltk.tokenize.sent_tokenize(
                            file_data["text"], language="portuguese"
                        )

                    file_data["text_tokens"].insert(0, "<start>")
                    file_data["text_tokens"].append("<end>")

                    file_data["token_count"] = len(file_data["text_tokens"])
                    token_count += file_data["token_count"]
                else:
                    file_data["text_tokens"] = []
                    file_data["token_count"] = 0

                files.append(file_data)

        return files, token_count

    def read_all_files(self, corpus_directory, files_quantity):
        files_names = [
            os.path.join(corpus_directory, file)
            for file in os.listdir(corpus_directory)
            if file.endswith(".json")
        ]

        if files_quantity != None:
            files_names = files_names[:files_quantity]

        print(f"Lendo {len(files_names)} arquivos.")
        files, token_count = self.read_files_chunk(files_names)
        self.files_processed.extend(files)
        self.total_token_count += token_count

        return self.files_processed

    def split_test_train(self, split_proportion):
        proportional_token_count = split_proportion * self.total_token_count

        train_data = []
        test_data = []

        count_tokens = 0

        shuffled_files_processed = self.files_processed
        random.seed(4)
        random.shuffle(shuffled_files_processed)

        print("Separando pedaços de teste e treino, aguarde...")
        for file in tqdm(shuffled_files_processed):
            if count_tokens + file["token_count"] <= proportional_token_count:
                train_data += file["text_tokens"]
                count_tokens += file["token_count"]
            else:
                test_data += file["text_tokens"]

        print(f"Tamanho de texto de teste: {len(train_data)}")
        print(f"Tamanho de texto de treino: {len(test_data)}")

        return train_data, test_data
