import json
from typing import Any

import pandas as pd
import torch
from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from main.config.app_constants import AppConstants
from main.config.logger import global_logger
from main.utils.column_utils import Features
from main.utils.data.dto.Feature import DependentFeature
from main.utils.data.dto.MlRowValue import MlRowValue
from main.utils.index import get_device

device = get_device()


class Tester:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.o: MlRowValue = None
        self.combined_data = None

        with open(AppConstants.DATA_MODEL_PATH, 'r') as file:
            self.o = MlRowValue.from_dict(json.load(file))
        self.add_missing_test_features(self.o, [Features.CATEGORY])

    def add_missing_test_features(self, o: MlRowValue, features: list[DependentFeature]):
        for feature in features or []:
            feature_analysis = o.features_dict[feature.name]
            f_labels = feature_analysis.sub_feature_labels

            feature_analysis.label_encoder_cat = LabelEncoder()
            feature_analysis.onehot_encoder_cat = OneHotEncoder(sparse_output=False)
            feature_analysis.integer_encoded_cat = feature_analysis.label_encoder_cat.fit_transform(f_labels)
            feature_analysis.onehot_encoded_cat = feature_analysis.onehot_encoder_cat.fit_transform(
                feature_analysis.integer_encoded_cat.reshape(-1, 1))

            # Create dictionaries for category and sub-category mapping
            feature_analysis.category_mapping = dict(zip(f_labels, feature_analysis.onehot_encoded_cat))
            feature_analysis.num_categories = len(f_labels)
            for l in f_labels:
                k = feature_analysis.sub_feature_dict[l]
                self.add_missing_test_features(k, feature.children)

    def view_data(self):
        return self.combined_data

    def prep_test_o(self, test_o: MlRowValue):
        test_o.clean_dataframe()
        x_predict = test_o.tokenize_predict_data(self.tokenizer)
        predict_input_ids = torch.tensor(x_predict, dtype=torch.long)
        predict_dataset = TensorDataset(predict_input_ids)
        predict_dataloader = DataLoader(predict_dataset, batch_size=1, shuffle=False)
        print("Length of predict_dataloader:", len(predict_dataloader))
        return predict_dataloader

    def recursive_batch_predictor(self,
                                  depth:int,
                                  train_o: MlRowValue,
                                  data: dict[str, Any],
                                  batch,
                                  features: list[DependentFeature]):
        for feature in features or []:
            cat_model = train_o.load_cat_model(feature)
            cat_model.to(device)
            input_ids = batch[0].to(device)
            with torch.no_grad():
                category_probs = cat_model(input_ids)
                category_predictions = category_probs.argmax(dim=-1)

            for i in range(input_ids.size(0)):
                # Unmap input_ids to the original description
                if depth == 0:
                    single_input_ids = input_ids[i].to('cpu')
                    tokens = self.tokenizer.convert_ids_to_tokens(single_input_ids)
                    description = self.tokenizer.convert_tokens_to_string(
                        [token for token in tokens if token != "[PAD]"]).strip()
                    data['Description'].append(description)

                predictions = train_o.features_dict[feature.name].label_encoder_cat.inverse_transform(
                    [category_predictions[i].item()])
                prediction = predictions[0]
                global_logger.info(f"""
                    Feature: {feature.name}
                    Predicted: {prediction}
                    Predictions: {predictions}
                    """)
                if feature.name not in data:
                    data[feature.name] = []
                data[feature.name].append(prediction)
                self.recursive_batch_predictor(
                    depth + 1,
                    train_o.features_dict[feature.name].sub_feature_dict[prediction],
                    data,
                    batch,
                    feature.children
                )

    def predict(self, df:DataFrame,callback=None, output_csv=None):
        train_o = self.o
        test_o = MlRowValue.test_from_df(df)

        # Load models

        predict_dataloader = self.prep_test_o(test_o)
        stats  = 0
        len_predict_dataloader = len(predict_dataloader)
        # Lists to store predictions and descriptions
        data = {
            'Description': [],
            'Category': [],
        }
        for batch in predict_dataloader:
            self.recursive_batch_predictor(0, train_o, data, batch, features=[Features.CATEGORY])
            stats +=1
            if callback:
                callback(int((stats/len_predict_dataloader)*100))

        predict_df = pd.DataFrame(data)

        if output_csv:
            # Write to CSV
            predict_df.to_csv(output_csv)
        return predict_df, output_csv


def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    trainer = Tester(tokenizer)
    data = {
        'Description': ['Tesla charger CALIFORNIA 800-Tesla CASZTUN69246602400','DOLLAR TREE LITTLE ROCK AR'],
        'Category': ["",""],
        'Sub_Category': ["",""]
    }

    df_to_process = pd.DataFrame(data)
    df, csv = trainer.predict(df=df_to_process)
    print(df)


if __name__ == '__main__':
    main()
