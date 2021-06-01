from abc import ABC, abstractmethod
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from joblib import dump, load


class Model(ABC):
    '''
    Model is an abstract class to define common methods across all models.
    All regression models we use should inherit from Model.
    A score can be retrieved by calling model.propagate()

    init args: data_path as a string. This is the ONLY input.

    '''

    def __init__(self, data_dir: str, train_file: str, test_file: str, feature_count=20):
        pass

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, model, X_test):
        pass

    @abstractmethod
    def rank_importance(self):
        pass

    @abstractmethod
    def save_model(self, filenames):
        pass


class ClassifierModel(Model):

    def __init__(self, data_dir: str, train_file: str, test_file: str, feature_count=20):
        super().__init__(data_dir, train_file, test_file)
        self.feature_count = feature_count
        self.data_dir = data_dir
        train_path = data_dir + '/' + train_file
        test_path = data_dir + '/' + test_file
        self.X_train, self.y_train, = self._import_data(train_path)
        self.X_test, self.y_test = self._import_data(test_path)
        self.y_pred = None

    def _import_data(self, file_path: str):
        data_set = pd.read_csv(file_path, index_col=0)
        X = data_set.iloc[:, :self.feature_count]
        y = data_set.iloc[:, -1]
        return X, y

    @abstractmethod
    def train(self, X_train, y_train):
        pass

    def predict(self, model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    def rank_importance(self):
        pass

    def calc_accuracy(self, y_true, y_pred):
        true_list = y_true - y_pred == 0
        acc = (sum(true_list == 1)) / len(true_list == 1)
        return acc

    def save_model(self, filename):
        dump(self.clf, self.data_dir + '/' + filename + '.joblib')

    def execute_all(self):
        self.clf = self.train(self.X_train, self.y_train)
        self.y_pred = self.predict(self.clf, self.X_test)
        accuracy = self.calc_accuracy(self.y_test, self.y_pred)
        print("Model accuracy is: " + str(accuracy))
        # self.save_model('logistic_v3')

class LogisticModel(ClassifierModel):

    def train(self, X_train, y_train):
        clf = LogisticRegression(random_state=0).fit(X_train, y_train)
        return clf

    def prediction_probs(self, model, X_test):
        y_probs = model.predict_proba(X_test)
        return y_probs


class SVMModel(ClassifierModel):

    def train(self, X_train, y_train):
        clf = svm.SVC()
        clf.fit(X_train, y_train)
        return clf










