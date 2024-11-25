from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.linear_model import SGDClassifier


class HyperparameterOptimizer:

    def __init__(self, input_data, label_data) -> None:
        self.input_data = input_data
        self.label_data = label_data

        self.vectorizer_parameters = {
            "vect__max_df": (0.6, 0.8, 1.0),
            "vect__min_df": (1, 3, 5, 10),
            # "vect__ngram_range": ((1, 2), (2, 3)),
            "vect__norm": ("l1", "l2"),
        }

        self.results = {"multi_nb": None, "logistic_regression": None}

        self.__split_data()
        self.__find_multi_nb_parameters()
        self.__find_logistic_regression_parameters()

    def __split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.input_data, self.label_data, test_size=0.2, random_state=42
        )

        print(f"Tamanho de treino: {X_train.shape}")
        print(f"Tamanho de teste: {X_test.shape}")
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def __find_multi_nb_parameters(self):
        print("Encontrando melhores parâmetros para Naive Bayes Multinomial")

        multi_nb_parameters = {
            "clf__alpha": [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
        }

        self.results["multi_nb"] = self.__apply_grid_search(
            multi_nb_parameters, MultinomialNB()
        )

        print("Finalizada busca por melhores parâmetros para Naive Bayes Multinomial")

    def __find_logistic_regression_parameters(self):
        print("Encontrando melhores parâmetros para Regressão Logística")

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

    def __apply_grid_search(self, model_parameters, classifier):

        params = model_parameters | self.vectorizer_parameters

        pipeline = Pipeline(
            [
                ("vect", TfidfVectorizer(stop_words="english")),
                ("clf", classifier),
            ]
        )

        print(f"\nLista de parâmetros utilizada: {params}\n")
        grid = GridSearchCV(pipeline, param_grid=params, n_jobs=-1, cv=2, verbose=2)

        grid.fit(self.X_train, self.y_train)

        return grid

    def get_results(self):
        return self.results

    def get_splitted_data(self):
        return self.X_train, self.X_test, self.y_train, self.y_test
