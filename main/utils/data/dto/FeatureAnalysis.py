from typing import Any, TYPE_CHECKING

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import importlib

if TYPE_CHECKING:
    from main.utils.data.dto.MlRowValue import MlRowValue



class FeatureAnalysis:

    def __init__(self):
        # Initialize instance attributes
        self.feature = None
        self.sub_feature_dict: dict[str,'MlRowValue'] = {}
        self.sub_feature_labels: list[str] = []
        self.label_encoder_cat: LabelEncoder = LabelEncoder()
        self.onehot_encoder_cat: OneHotEncoder = OneHotEncoder()
        self.onehot_encoder_subcat: OneHotEncoder = OneHotEncoder()
        self.integer_encoded_cat: Any = None
        self.onehot_encoded_cat: Any = None

        self.category_to_onehot_encoded_mapping: dict[str, int] = {}
        self.int_encoded_mapping_to_category: dict[int, str] = {}
    def to_dict(self):

        return {
            "sub_feature_dict": self.sub_feature_dict,
            "sub_feature_labels": self.sub_feature_labels,
            "label_encoder_cat": str(type(self.label_encoder_cat)),  # Just store type for now
            "onehot_encoder_cat": str(type(self.onehot_encoder_cat)),  # Just store type for now
            "onehot_encoder_subcat": str(type(self.onehot_encoder_subcat)),  # Just store type for now
            "integer_encoded_cat": str(type(self.integer_encoded_cat)),
            "onehot_encoded_cat": str(type(self.onehot_encoded_cat)),
            "category_to_onehot_encoded_mapping": str(type(self.category_to_onehot_encoded_mapping)),
            "int_encoded_mapping_to_category": str(type(self.int_encoded_mapping_to_category)),

        }
    @staticmethod
    def from_dict(data: dict) -> 'FeatureAnalysis':
        MlRowValue_module = importlib.import_module("main.utils.data.dto.MlRowValue")
        MlRowValue = getattr(MlRowValue_module, "MlRowValue")
        obj = FeatureAnalysis()
        obj.feature = data.get("feature")
        obj.sub_feature_dict = {k: MlRowValue.from_dict(v) for k, v in data.get("sub_feature_dict", {}).items()}
        obj.sub_feature_labels = data.get("sub_feature_labels", [])
        obj.category_to_onehot_encoded_mapping = data.get("category_to_onehot_encoded_mapping", {})
        obj.int_encoded_mapping_to_category = data.get("int_encoded_mapping_to_category", {})
        return obj
