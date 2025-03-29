import os

import matplotlib
from transformers import BertTokenizer

from main.ml.test import Tester
# custom modules
from main.ml.train import Trainer
from main.utils.dump_utils import dump_o
from main.utils.index import get_device

matplotlib.use('Agg')

device = get_device()

class MlModel:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    __instance = None

    def __init__(self):
        self._trainer = None
        self._tester = None
        self.__instance = self

    @staticmethod
    def default():
        if not MlModel.__instance:
            MlModel.__instance = MlModel()
        return MlModel.__instance

    def get_trainer(self):

        learning_rate = 1e-5
        epochs = 2
        if not self._trainer:
            self._trainer = Trainer(
                tokenizer=self.tokenizer,
                learning_rate=learning_rate,
                epochs=epochs)
        return self._trainer

    def get_tester(self):
        if not self._tester:
            self._tester = Tester(
                tokenizer=self.tokenizer
            )
        return self._tester


def main():
    # trainer = MlModel.default().get_trainer()
    # # Execute & Save Models
    # o = MlRowValue.train_from_folder(AppConstants.TRAIN_PATH)
    # trainer.execute_cat_model(o)
    # dump_o(trainer.data_preprocessor.all_data)
    tester = MlModel.default().get_tester()



if __name__ == '__main__':
    main()
