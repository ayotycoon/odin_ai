import json
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from main.ml.MyBertModel import MyBertModel
from main.utils.column_utils import Features
from main.config.app_constants import AppConstants
from main.config.logger import global_logger, clear_console
from main.utils.cacheable import cacheable_callback
from main.utils.data.dto.Feature import DependentFeature
from main.utils.data.dto.MlRowValue import MlRowValue
from main.utils.dump_utils import dump_o
from main.utils.index import get_device
from main.utils.log_time import log_time, ElapsedTime
from main.utils.path_utils import safe_access_path

device = get_device()

class TrainFeatureInstance:
    def __init__(self, cat_model=None, cat_train_dataloader=None, cat_val_dataloader=None,
                 num=None, int_encoded_mapping_to_category=None, labels=None, folder_path = None):
        self.cat_model = cat_model
        self.cat_train_dataloader = cat_train_dataloader
        self.cat_val_dataloader = cat_val_dataloader
        self.num = num
        self.int_encoded_mapping_to_category = int_encoded_mapping_to_category
        self.labels = labels

    def __repr__(self):
        return (f"FeatureInstance(cat_model={self.cat_model}, "
                f"cat_train_dataloader={self.cat_train_dataloader}, "
                f"cat_val_dataloader={self.cat_val_dataloader}, "
                f"num={self.num}, "
                f"int_encoded_mapping_to_category={self.int_encoded_mapping_to_category}, "
                f"labels={self.labels})")


class Trainer:
    def __init__(self,
                 tokenizer: Union[bool, BertTokenizer],
                 learning_rate: float,
                 epochs: int,
                 batch_size=64,
                 no_improvement_patience: int = 5,

                 ):
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.no_improvement_patience = no_improvement_patience
        self.general_elapsed_tracker = ElapsedTime(replace=False)

    def trainer_init(self, o:MlRowValue, features: list[DependentFeature]):
        o.clean_dataframe()
        o.tokenize_data(self.tokenizer)

        # (x_train, x_test, y_cat_train, y_cat_test, y_sub_train, y_sub_test) = o.prepare_data(path)
        f = o.prepare_data()
        df = o.df
        x_train = f[0]
        x_test = f[1]
        # Convert tokenized sequences to input IDs
        train_input_ids = torch.tensor(x_train)
        val_input_ids = torch.tensor(x_test)
        global_logger.info(f"Total number of data points: {df.shape[0]} "
                           f"Number of training data points: {len(x_train)} "
                           f"Number of testing data points: {len(x_test)} ")



        s = 2
        mapped: dict[str, TrainFeatureInstance] = {}
        for index,feature in enumerate(features):

            feature_train = f[s+index]
            feature_test = f[s+index+1]
            s +=2
            labels = o.features_dict[feature.name].sub_feature_labels
            num = len(labels)
            if  not num:
                continue

            y_cat_train = np.array(feature_train)
            y_cat_test = np.array(feature_test)
            y_cat_train = np.argmax(y_cat_train, axis=1)
            y_cat_test = np.argmax(y_cat_test, axis=1)
            y_cat_train = torch.tensor(y_cat_train, dtype=torch.long)
            y_cat_test = torch.tensor(y_cat_test, dtype=torch.long)
            cat_train_dataset = TensorDataset(train_input_ids, y_cat_train)
            cat_train_dataloader = DataLoader(cat_train_dataset, batch_size=self.batch_size, shuffle=True)
            cat_val_dataset = TensorDataset(val_input_ids, y_cat_test)
            cat_val_dataloader = DataLoader(cat_val_dataset, batch_size=self.batch_size, shuffle=False)
            global_logger.info(f"{feature.name} Number of training batches: {len(cat_train_dataloader)}")
            global_logger.info(f"{feature.name} Number of validation batches: {len(cat_val_dataloader)}")
            cat_model = MyBertModel(num)
            int_encoded_mapping_to_category = o.features_dict[
                feature.name].int_encoded_mapping_to_category
            instance = TrainFeatureInstance(
                cat_model=cat_model,
                cat_train_dataloader=cat_train_dataloader,
                cat_val_dataloader=cat_val_dataloader,
                num=num,
                int_encoded_mapping_to_category=int_encoded_mapping_to_category,
                labels=labels)

            mapped[feature.name] = instance

        return mapped

    @log_time
    def train_model(self, o: MlRowValue,model: MyBertModel, feature: DependentFeature, train_dataloader, val_dataloader, folder_path):
        optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
        category_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        model = model.to(device)

        best_val_loss = float('inf')
        no_improvement_epochs = 0
        feature_elapsed_tracker = ElapsedTime(replace=False)
        batch_elapsed_tracker = ElapsedTime(replace=True)
        all_preds = []
        all_labels = []
        train_dataloader_len = len(train_dataloader)
        batch_print_interval =max(min(200, int(train_dataloader_len/3)),1)
        for epoch in range(self.epochs):  # 1 indexed
            model.train()
            total_train_loss = 0
            correct_train = 0

            self.batch_logger(
                o = o,
                avg_train_loss=0,
                avg_val_loss=0,
                epoch=epoch,
                batch_i=-1,
                batch_elapsed_tracker=batch_elapsed_tracker,
                feature=feature,
                feature_elapsed_tracker=feature_elapsed_tracker,
                train_acc=0,
                train_dataloader_len=train_dataloader_len,
                val_acc=0,
                batch_print_interval=0,
                folder_path=folder_path
            )

            for batch_i, batch in enumerate(train_dataloader):
                input_ids, y_feature = [item.to(device) for item in batch[:2]]
                optimizer.zero_grad()
                feature_probs = model(input_ids)
                cat_loss = category_loss_fn(feature_probs, y_feature)
                total_train_loss += cat_loss.item()
                correct_train += (feature_probs.argmax(dim=1) == y_feature).sum().item()
                cat_loss.backward()
                optimizer.step()


                is_regular_interval = (batch_i + 1) % batch_print_interval == 0
                is_last_batch = train_dataloader_len == batch_i + 1
                if is_regular_interval or is_last_batch:
                    # Validation phase inside the batch loop
                    model.eval()
                    total_val_loss = 0
                    correct_val = 0

                    with torch.no_grad():
                        for batch in val_dataloader:
                            input_ids, y_feature = [item.to(device) for item in batch[:2]]

                            feature_probs = model(input_ids)
                            cat_loss = category_loss_fn(feature_probs, y_feature)
                            total_val_loss += cat_loss.item()
                            preds = feature_probs.argmax(dim=1)
                            correct_val += (preds == y_feature).sum().item()
                            # Store for confusion matrix
                            all_preds.extend(preds.cpu().numpy())  # Convert to numpy
                            all_labels.extend(y_feature.cpu().numpy())

                    # Compute Validation Metrics
                    avg_val_loss = total_val_loss / len(val_dataloader)
                    val_acc = correct_val / len(val_dataloader.dataset)
                    avg_train_loss = total_train_loss / (batch_i + 1)  # current average for this epoch up to batch i
                    train_acc = (correct_train / ((batch_i + 1) * len(batch))) / 100

                    self.batch_logger(
                        o = o,
                        avg_train_loss=avg_train_loss,
                        avg_val_loss=avg_val_loss,
                        epoch=epoch,
                        batch_i=batch_i,
                        batch_elapsed_tracker=batch_elapsed_tracker,
                        feature=feature,
                        feature_elapsed_tracker=feature_elapsed_tracker,
                        train_acc=train_acc,
                        train_dataloader_len=train_dataloader_len,
                        val_acc=val_acc,
                        batch_print_interval=batch_print_interval,
                        folder_path=folder_path
                    )
                    model.train()
            history['train_loss'].append(avg_train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            # Update learning rate
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_epochs = 0
            else:
                no_improvement_epochs += 1
            if no_improvement_epochs >= self.no_improvement_patience:
                print(f"Stopping early due to no improvement after {self.no_improvement_patience} epochs.")
                break

        return history, model, all_labels, all_preds

    def batch_logger(self,o:MlRowValue,
                     avg_train_loss: int,
                     avg_val_loss: int,
                     epoch: int,
                     batch_i: int,
                     batch_elapsed_tracker: ElapsedTime,
                     feature: DependentFeature,
                     feature_elapsed_tracker: ElapsedTime,
                     train_acc: int,
                     train_dataloader_len:int,
                     batch_print_interval:int,
                     val_acc: int,folder_path:str):
        clear_console()
        global_logger.info(
            f"""
                        name: {o.name}
                        folder_path: {folder_path}
                        feature: {feature.path}
                        Epoch {epoch + 1}/{self.epochs} - Batch {batch_i + 1}/{train_dataloader_len}
                        batch_print_interval: {batch_print_interval}
                        Training loss: {avg_train_loss:.4f}, Training Acc: {train_acc:.4f}
                        Validation loss: {avg_val_loss:.4f}, Validation Acc: {val_acc:.4f}
                        batch_time: {batch_elapsed_tracker.log()}
                        feature_time: {feature_elapsed_tracker.log()}
                        total_time: {self.general_elapsed_tracker.log()}
                        """
        )

    @log_time
    def draw_heatmap(self, feature: DependentFeature, conf_matrix, tick_labels, export_path):

        if feature.name == 'Category':
            plt.figure(figsize=(10, 10))
        else:  # If it's a subcategory model, combine category and subcategory labels
            plt.figure(figsize=(30, 20))

        sns.heatmap(conf_matrix,
                    annot=True,
                    fmt='d',
                    xticklabels=tick_labels,
                    yticklabels=tick_labels
                    )
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt_path = safe_access_path(export_path)
        plt.savefig(safe_access_path(plt_path), dpi=300, bbox_inches='tight', pad_inches=0.1)
        global_logger.info("heatmap saved")
        plt.close()

    def plot_training_history(self, plt_path, history):
        expected_keys = ['train_loss', 'train_acc', 'val_loss', 'val_acc']
        for key in expected_keys:
            if key not in history.keys():
                print(f"Error: Expected key {key} not found in history")
                return

        # Plot training and validation loss
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.legend()
        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training vs Validation Accuracy')
        plt.legend()
        plt.tight_layout()

        plt.savefig(safe_access_path(plt_path), dpi=300, bbox_inches='tight')  # Adjust `dpi` for resolution

        global_logger.info("Training vs Validation Loss saved")
        plt.close()

    def execute_model(self,
                      o: MlRowValue,
                      model,
                      feature: DependentFeature,
                      train_dataloader,
                      val_dataloader,
                      folder_path,
                      label_mapping,
                      feature_list,
                      cache_key
                      ):
        o.data_dict['model_path'] = f"{folder_path}/modelV1.pt"
        o.data_dict['t_vs_a'] = f'{folder_path}/Training_vs_Validation_Accuracy.png'
        o.data_dict['heat_map'] = f'{folder_path}/heat_map.png'
        '''Generalized Model Training & Saving'''
        model.to(device)

        history, trained_model, all_labels, all_preds = cacheable_callback(
            cache_key,
            lambda: self.train_model(
                o = o,
                model=model,
                feature=feature,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                folder_path=folder_path,
            )
        )
        model.to('cpu')

        torch.save(model.state_dict(), safe_access_path(o.data_dict['model_path']))

        self.plot_training_history(o.data_dict['t_vs_a'], history)

        conf_matrix = confusion_matrix(all_labels, all_preds)
        xlabels = [label_mapping[x] for x in np.unique(all_labels)]
        self.draw_heatmap(feature, conf_matrix, xlabels,o.data_dict['heat_map'])

        diff = list(
            set(feature_list) - set(xlabels))
        o.data_dict['missing_features'] = diff
        o.data_dict['data_path'] = f'{folder_path}/missing_features.json'
        self.log_training_result(o.data_dict['data_path'], o.data_dict)

    @log_time
    def execute_cat_model(self, o:MlRowValue = None, features=[Features.CATEGORY], folder_path:str=None):
        if not o:
            o = MlRowValue.train_from_folder(AppConstants.TRAIN_PATH)
        if not folder_path:
            folder_path = f"{AppConstants.ML_EXPORT_FOLDER_PATH}/{o.name}"
            self.general_elapsed_tracker.reset()
        if not features:
            return

        mapped = self.trainer_init(o=o, features=features)
        cache_key = f"train_model_{folder_path.replace(AppConstants.ML_EXPORT_FOLDER_PATH,'').replace('/','_')}"
        for feature in features:
            k = mapped[feature.name]

            self.execute_model(
                o,
                k.cat_model,
                feature,
                k.cat_train_dataloader,
                k.cat_val_dataloader,
                f'{folder_path}',
                k.int_encoded_mapping_to_category,
                k.labels,
                cache_key
            )
            labels = o.features_dict[feature.name].sub_feature_labels
            for label in labels:
                j = o.features_dict[feature.name].sub_feature_dict[label]
                self.execute_cat_model(j, feature.children,f"{folder_path}/{feature.name}/{j.name}")




    def log_training_result(self, path: str, obj):
        with open(path, "w") as json_file:
            json.dump(obj, json_file, indent=4)  # `indent=4` makes it readable


def main():

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    learning_rate = 1e-5
    epochs = 2
    trainer = Trainer(tokenizer,learning_rate,epochs)
    # Execute & Save Models
    o = MlRowValue.train_from_folder(AppConstants.TRAIN_PATH)
    trainer.execute_cat_model(o)
    dump_o(o)




if __name__ == '__main__':
    main()
