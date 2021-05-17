from abc import ABC, abstractmethod
import sklearn


class RegressionModel(ABC):
    '''
    RegressionModel is an abstract class to define common methods across all models.
    All regression models we use should inherit from RegressionModel.
    A score can be retrieved by calling model.propagate()

    init args: data_path as a string. This is the ONLY input.

    '''

    @abstractmethod
    def __init__(self, data_path: str):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def propagate(self):
        pass

    @abstractmethod
    def rank_importance(self):
        pass


class ElasticNetModel(RegressionModel):

    def __init__(self, data_path: str):


