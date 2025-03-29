import random
import re

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
import torch
from transformers import BertTokenizer

from main.config.app_constants import AppConstants
from main.config.logger import global_logger
from main.ml.MyBertModel import MyBertModel
from main.utils.column_utils import FeatureValue, Features, DependentFeature, get_structure
from main.utils.data.dto.FeatureAnalysis import FeatureAnalysis
from main.utils.dump_utils import dump_o
from main.utils.index import get_all_files_recursive

nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 50)

stop_words = set(stopwords.words('english'))



def clean_dataframe(df: DataFrame):
    df["Category"].fillna("", inplace=True)
    df["Sub_Category"].fillna("", inplace=True)
    # Remove any duplicate rows where Description and Category are the same
    df.drop_duplicates(subset=['Description', 'Category'], keep='first', inplace=True)
    # Use regular expression to remove non-letter characters
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z ]', '', str(x)))
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'[^a-zA-Z\-\' ]', '', x))
    # remove white spaces and make all lowercase
    df['Description'] = df['Description'].apply(lambda x: x.strip().lower())
    # Replace multiple spaces with a single space
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'\s+', ' ', x))
    # Remove any numbers from the description
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'\d+', '', x))
    # Remove the word 'and', 'the' from the description
    df['Description'] = df['Description'].apply(
        lambda x: re.sub(r'\b(and|the|set|inch|of|in|made|to|by|compatible|with|for|set|other|cm|st|street|ave)\b', '',
                         x))
    # Make all letters lowercase in Description column
    df['Description'] = df['Description'].str.lower()
    # Remove any single letter words from Description column
    df['Description'] = df['Description'].apply(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x))
    # Remove stop words from Description column using NLTK
    df['Description'] = df['Description'].apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
    # self.df.dropna(subset=['Category', 'Sub_Category'], inplace=True)
    return df

class MlRowValue:
    df: DataFrame
    features_dict: dict[str, 'FeatureAnalysis']
    data_dict: dict[str,str]

    def __init__(self, name: str, type: str):
        self.name = name
        self.type = type
        self.df = None
        self.features_dict:dict[str, 'FeatureAnalysis'] = {}
        self.data_dict = {}

    @staticmethod
    def from_dict(dict_data: dict) -> 'MlRowValue':
        obj = MlRowValue(dict_data["name"], dict_data["type"])

        # Placeholder: You may need logic to convert df from string if needed
        obj.df = None  # since df is stored as a string in JSON

        obj.features_dict = {
            k: FeatureAnalysis.from_dict(v) for k, v in dict_data.get("features_dict", {}).items()
        }

        obj.data_dict = dict_data.get("data_dict", {})
        return obj

    def __repr__(self):
        return f"DataRowValue(name={self.name}, type={self.type})"

    def to_dict(self):
        return {

            "name": self.name,
            "type": self.type,
            "df": str(type(self.df)),
            "features_dict": self.features_dict,
            "data_dict": self.data_dict,
        }


    @staticmethod
    def train_from_df(df: DataFrame):

        def def_processor( row_value: 'MlRowValue', features: list[DependentFeature], structure: list[FeatureValue], folder_path):

            # row_value.df.to_csv(f"{safe_access_path(folder_path)}/trained.csv")
            for feature in features or []:
                sub_feature_dict = [MlRowValue(f.name, f.type) for f in structure or []]
                feature_analysis = FeatureAnalysis()
                feature_analysis.sub_feature_dict = {v.name: v for v in sub_feature_dict}

                row_value.features_dict[feature.name] = feature_analysis

                feature_analysis = row_value.features_dict[feature.name]
                feature_analysis.sub_feature_labels = [s.name for s in sub_feature_dict]
                if len(feature_analysis.sub_feature_labels) == 0:
                    continue
                # Convert categorical variables to numerical labels
                feature_analysis.label_encoder_cat = LabelEncoder()
                feature_analysis.onehot_encoder_cat = OneHotEncoder(sparse_output=False)
                # Encode category_list using label_encoder_cat
                feature_analysis.integer_encoded_cat = feature_analysis.label_encoder_cat.fit_transform(
                    feature_analysis.sub_feature_labels)
                feature_analysis.onehot_encoded_cat = feature_analysis.onehot_encoder_cat.fit_transform(
                    feature_analysis.integer_encoded_cat.reshape(-1, 1))
                feature_analysis.category_to_onehot_encoded_mapping = dict(
                    zip(feature_analysis.sub_feature_labels, feature_analysis.onehot_encoded_cat))
                feature_analysis.int_encoded_mapping_to_category = dict(
                    zip(feature_analysis.integer_encoded_cat, feature_analysis.sub_feature_labels))
                feature_analysis.num_categories = len(feature_analysis.sub_feature_labels)
                feature_analysis.category_list = feature_analysis.sub_feature_labels

                if not feature.children:
                    continue
                df_dict: dict[str, DataFrame] = {}
                grouped = row_value.df.groupby(feature.name)
                for row_name, group_df in grouped:
                    df_dict[row_name] = group_df
                for index, j in enumerate(sub_feature_dict):
                    if j.name not in df_dict:
                        continue
                    j.df = df_dict[j.name]
                    def_processor(j, feature.children, structure[index].subs, f"{folder_path}/{feature.path.replace('|','/')}/{j.name}")

        o = MlRowValue('main', 'main')
 
        o.df = df
        # row_value.df.drop(row_value.df[~row_value.df[feature.name].isin(feature_analysis.sub_feature_dict.keys())].index, inplace=True)
        df.dropna(inplace=True)
        def_processor(o, [Features.CATEGORY], get_structure(), f"{AppConstants.ML_EXPORT_FOLDER_PATH}/{o.name}")
        dump_o(o)
        return o

    @staticmethod
    def train_from_folder(folder_path: str):
        all_file_pd = None

        file_paths = get_all_files_recursive(folder_path)
        for i, file_path in enumerate(file_paths):
            global_logger.info(f"Importing dataset: {file_path}")

            # Read the first file with headers, subsequent ones without headers
            file_pd = pd.read_csv(file_path, header=0)

            # Concatenate DataFrames instead of merging
            if all_file_pd is None:
                all_file_pd = file_pd
            else:
                all_file_pd = pd.concat([all_file_pd, file_pd], ignore_index=True)

 
        return MlRowValue.train_from_df(all_file_pd)

    @staticmethod
    def test_from_df(df: DataFrame):
        o = MlRowValue('test', 'test')
        o.df =df

        return o

    # def tokenize_predict_data(self, max_len=105, tokenizer = None):
    #     # Tokenize the 'Description' column
    #     tokenized_desc = [tokenizer.tokenize(text) for text in self.df['Description']]
    #     self.df['Tokenized'] = tokenized_desc
    #     # Convert the tokenized sequences to IDs
    #     tokenized_desc_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_desc]
    #     self.df['Tokenized_ids'] = tokenized_desc_ids
    #     # Find the maximum length of the tokenized sequences
    #     actual_max = max([len(tokens) for tokens in tokenized_desc_ids])
    #     print("Max length in the data:", actual_max)
    #     # Pad the sequences to the max_len
    #     padded_desc = tf.keras.preprocessing.sequence.pad_sequences(tokenized_desc_ids, maxlen=max_len,
    #                                                                 padding='post', truncating='post')
    #     X = self.df['Tokenized_padded'] = np.array(padded_desc).tolist()
    #     return X
    #
    # def clean_dataframe(self):
    #     return clean_dataframe(self.df)

    def clean_dataframe(self):
        return clean_dataframe(self.df)
    def tokenize_data(self, tokenizer:BertTokenizer,max_len=50):

        df = self.df
        o = self

        # Tokenize the 'Description' column
        tokenized_desc = [tokenizer.tokenize(text) for text in df['Description']]
        df['Tokenized'] = tokenized_desc
        # Convert the tokenized sequences to IDs
        tokenized_desc_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_desc]
        df['Tokenized_ids'] = tokenized_desc_ids
        # Find the maximum length of the tokenized sequences
        actual_max = max([len(tokens) for tokens in tokenized_desc_ids])
        print("Max length in the data:", actual_max)
        # Pad the sequences to the max_len
        padded_desc = tf.keras.preprocessing.sequence.pad_sequences(tokenized_desc_ids, maxlen=max_len,
                                                                    padding='post', truncating='post')
        df['Tokenized_padded'] = np.array(padded_desc).tolist()

        def category_to_onehot(_x, k):
            _dict = o.features_dict[k].category_to_onehot_encoded_mapping
            if _x not in _dict:
                raise Exception(f"{_x} does not exist in mapping for {k}")
            return _dict.get(_x)
        for k in o.features_dict.keys():
            # Map the Category and Sub_Category values to the corresponding one-hot encoded vectors

            df[f'Tok_{k}'] = df[k].apply(category_to_onehot, k=k)
        return df

    def prepare_data(self):
        df = self.df
        o = self
        # Separate Category and Sub_Category labels
        g = []
        for k in o.features_dict.keys():
            # Map the Category and Sub_Category values to the corresponding one-hot encoded vectors
            y_cat = df[f'Tok_{k}'].tolist()
            g.append(y_cat)
        X = df["Tokenized_padded"].tolist()
        f = train_test_split(X, *g, test_size=0.33, random_state=42)
        # X_train, X_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test = f
        return f

    def predict_prepare_data(self):

        df = self.df
        df = df.dropna(subset=["Sub_Category", "Category"])
        # Separate Category and Sub_Category labels
        X = df["Tokenized_padded"].tolist()
        return X

    def shuffle_sentences(self):

        df = self.df
        """Double the dataset by shuffling the words in the Description."""
        augmented_df = df.copy()
        # Shuffle the words in each Description
        augmented_df['Description'] = augmented_df['Description'].apply(
            lambda desc: ' '.join(random.sample(desc.split(), len(desc.split()))))
        # Concatenate the original and augmented dataframes
        df = pd.concat([df, augmented_df], ignore_index=True)
        return df

    def load_cat_model(self, feature:DependentFeature):
        cat_model = MyBertModel(len(self.features_dict[feature.name].sub_feature_labels))
        s = self.data_dict['model_path']
        state_dict = torch.load(s)
        cat_model.load_state_dict(state_dict)
        cat_model.eval()
        return cat_model

    def tokenize_predict_data(self, tokenizer:BertTokenizer, max_len=105):
        # Tokenize the 'Description' column
        tokenized_desc = [tokenizer.tokenize(text) for text in self.df['Description']]
        self.df['Tokenized'] = tokenized_desc
        # Convert the tokenized sequences to IDs
        tokenized_desc_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokenized_desc]
        self.df['Tokenized_ids'] = tokenized_desc_ids
        # Find the maximum length of the tokenized sequences
        actual_max = max([len(tokens) for tokens in tokenized_desc_ids])
        print("Max length in the data:", actual_max)
        # Pad the sequences to the max_len
        padded_desc = tf.keras.preprocessing.sequence.pad_sequences(tokenized_desc_ids, maxlen=max_len,
                                                                    padding='post', truncating='post')
        X = self.df['Tokenized_padded'] = np.array(padded_desc).tolist()
        return X

