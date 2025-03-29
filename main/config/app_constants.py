import json
import os
from enum import Enum
from typing import Union

from main.utils.data.dto.Feature import DependentFeature
from main.utils.path_utils import safe_access_path








class AppVersion:
    __version = 'v2'
    @staticmethod
    def is_v2():
        return AppVersion.__version == 'v2'
    @staticmethod
    def is_v1():
        return AppVersion.__version == 'v1'




class AppConstants:

    APP_PATH = os.getcwd()
    FEATURES_CSV_PATH = 'data/features/categories.csv'
    COLUMN_STRUCTURE_PATH = '.temp/columns_structure.json'
    TRAIN_PATH = '.temp/data/csv'
    DATA_MODEL_PATH = '.temp/data_model.json'
    LOG_FOLDER_PATH = safe_access_path(f'{os.getcwd()}/.temp/logs')
    # ML_EXPORT_FOLDER_PATH =  safe_access_path(f'{os.getcwd()}/.temp/exports')
    ML_EXPORT_FOLDER_PATH =  safe_access_path('.temp/exports')
    version = AppVersion
