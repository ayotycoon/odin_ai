import os
from typing import List

import pandas as pd

from main.config.app_constants import AppConstants
from main.utils.data.dto.Feature import IndependentFeature, DependentFeature
from main.utils.data.dto.FeatureValue import FeatureValue
from main.utils.dump_utils import dump_c




DESCRIPTION = IndependentFeature('Description')

SUB_CATEGORY = DependentFeature('Sub_Category')
CATEGORY = DependentFeature('Category', [SUB_CATEGORY])


class Features:
    DESCRIPTION = DESCRIPTION
    CATEGORY = CATEGORY

features_df = pd.read_csv(AppConstants.FEATURES_CSV_PATH)

columns_structure: List[FeatureValue] = []
# Grouping by Category
grouped = features_df.groupby('Category')['Sub_Category'].apply(list).reset_index()
for _, row in grouped.iterrows():
    subs = [FeatureValue(x, '') for x in row['Sub_Category']]
    columns_structure.append(FeatureValue(row['Category'], '', subs))
f = ''
def get_structure():
    return columns_structure


def get_features_list():
    return [CATEGORY]

dump_c(columns_structure)
