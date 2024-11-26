import re
import pandas as pd
import nltk

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

from nltk.stem import WordNetLemmatizer

nltk.download("wordnet")  # Resgatando dados para lematização


class HyperparameterOptimizer:
    """
    Classe contendo o otimizador para os dados de texto recebidos.
    Contém funções de pré-processamento dos dados de texto, divisão de dados e aplicação de GridSearch

    Parâmetros:
    - input_data: Dados de entrada que serão utilizados (X).
    - label_data: Dados de saída que serão utilizados (Y)
    """

    def __init__(self, input_data, label_data) -> None:
        self.input_data = input_data
        self.label_data = label_data

        # Parâmetros do vetorizador que serão utilizados em todos os treinamentos
        self.vectorizer_parameters = {
            "vect__max_df": (0.6, 0.8),
            "vect__min_df": (1, 3, 5),
            # "vect__ngram_range": ((1, 2), (2, 3)),
            "vect__norm": ("l1", "l2"),
        }

        self.results = {"multi_nb": None, "logistic_regression": None, "sgdc": None}

        self.__preprocess_text_input()
        self.__split_data()

        self.__find_multi_nb_parameters()
        self.__find_logistic_regression_parameters()
        self.__find_sgdc_parameters()

    def __preprocess_text_input(self):
        """
        Função para pré-processar dados de texto de entrada.
        Realiza o apply para aplicar em todos os valores da Série do Dataframe a função de limpeza

        Irá retirar do texto todos os caracteres especiais, transformá-los em caixa baixa
        Por fim irá lematizar os textos de cada registro da série, salvando-o na classe
        """

        lemmatizer = WordNetLemmatizer()

        def clean_text(text):
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            text = text.lower()
            lemmed_text = lemmatizer.lemmatize(text)

            return lemmed_text

        self.input_data = self.input_data.apply(clean_text)

    def __split_data(self):
        """
        Função para dividir dados de treino e teste, a ser aplicada anterior a vetorização.
        Divide com 80% para treino e 20% para teste e salva os resultados na classe.
        """

        X_train, X_test, y_train, y_test = train_test_split(
            self.input_data, self.label_data, test_size=0.2, random_state=42
        )

        print(f"Tamanho de treino: {X_train.shape}")
        print(f"Tamanho de teste: {X_test.shape}")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __apply_grid_search(self, model_parameters, classifier):
        """
        Cria um pipeline contendo a vetorização e o modelo repassado como parâmetro.
        Aplica GridSearchCV para buscar os melhores hiperparâmetros do modelo dada a lista pré-selecionada.
        Aplicamos o fit do GridSearch apenas nos dados de treino.

        Parâmetros:
        - model_parameter: Dicionário contendo a lista de hiperparâmetros que iremos iterar.
        - classifier: Objeto contendo o modelo classificador que será analisado

        """

        # Realiza a junção de dicionários dos hiperparâmetros do vetorizador e modelo
        params = model_parameters | self.vectorizer_parameters

        # Cria um pipeline de processamento com o classificador recebido
        pipeline = Pipeline(
            [
                ("vect", TfidfVectorizer(stop_words="english")),
                ("clf", classifier),
            ]
        )

        print(f"\nLista de parâmetros utilizada: {params}\n")

        # Aplica o GridSearch utilizando todos os núcleos da máquina (n_jobs) e aplica
        # cross validation apenas 2 vezes para evitar um longo tempo de treinamento
        # O verbose foi escolhido para visualizarmos quantos fits serão necessários para
        # o GridSearch ser concluído.
        grid = GridSearchCV(pipeline, param_grid=params, n_jobs=-1, cv=2, verbose=2)

        grid.fit(self.X_train, self.y_train)

        return grid

    def __find_multi_nb_parameters(self):
        """
        Cria um modelo Multinomial NaiveBayes e o envia para ser analisado pelo GridSearch.
        """

        print("Encontrando melhores parâmetros para Naive Bayes Multinomial")

        # Alpha é um parâmetro de suavização do modelo
        multi_nb_parameters = {
            "clf__alpha": [0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        }

        # Salva os resultados do Grid no dicionário correspondente
        self.results["multi_nb"] = self.__apply_grid_search(
            multi_nb_parameters, MultinomialNB()
        )

        print("Finalizada busca por melhores parâmetros para Naive Bayes Multinomial")

    def __find_logistic_regression_parameters(self):
        """
        Cria um modelo de regressão logística e o envia para ser analisado pelo GridSearch.
        """

        print("Encontrando melhores parâmetros para Regressão Logística")

        # C controla a força da regularização aplicada ao modelos
        # O solver se refere ao algoritmo utilizado na otimização
        # Os demais hiperparâmetros não tiveram influência no resultado
        # sendo omitidos para melhorar eficiência
        logistic_regression_parameters = {
            "clf__C": [0.001, 0.1, 1, 10, 100],
            "clf__solver": ["lbfgs", "liblinear", "newton-cg"],
            # "clf__max_iter": [100, 200, 300],
            # "clf__penalty": ["l1", "l2"],
        }

        self.results["logistic_regression"] = self.__apply_grid_search(
            logistic_regression_parameters, LogisticRegression()
        )

        print("Finalizada busca por melhores parâmetros para Regressão Logística")

    def __find_sgdc_parameters(self):
        """
        Cria um modelo de classificado SGD e o envia para ser analisado pelo GridSearch.
        O algoritmo SGD em sua configuração padrão é uma aplicação do modelo SVM com um
        gradiente descente estocástico aplicado no aprendizado
        """
        print("Encontrando melhores parâmetros para SGDC")

        # Alpha é a constante que multiplica o termo de regularização
        # Quanto maior o valor, mais forte ela será
        sgdc_parameters = {
            # "clf__loss": ["hinge", "log_loss", "modified_huber"],
            # "clf__penalty": ["l2", "l1", "elasticnet"],
            "clf__alpha": [0.0001, 0.001, 0.01, 0.1],
        }
        self.results["sgdc"] = self.__apply_grid_search(
            sgdc_parameters, SGDClassifier(random_state=1)
        )

        print("Finalizada busca por melhores parâmetros para SGDC")

    def export_grid_results(self, path):
        """
        Exporta os resultados do grid para um arquivo CSV.

        Parâmetros:
        - path: caminho da pasta raiz que será salvo o resultado
        """
        for algorithm_results in self.results:
            grid_df = pd.DataFrame(self.results[algorithm_results].cv_results_)
            grid_df.to_csv(f"{path}/{algorithm_results}_results.csv")

    def get_results(self):
        return self.results

    def get_splitted_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
