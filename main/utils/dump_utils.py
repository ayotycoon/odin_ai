import json
from typing import TYPE_CHECKING

from main.config.app_constants import AppConstants
from main.utils.path_utils import safe_access_path

if TYPE_CHECKING:
    from main.utils.data.dto.MlRowValue import MlRowValue

def custom_serializer(obj):
    if hasattr(obj, 'to_dict'):
        return obj.to_dict()  # Call to_dict method to convert to a dictionary
    raise TypeError(f"Type {type(obj)} not serializable")

def dump_o(o:'MlRowValue'):
    with open(safe_access_path(AppConstants.DATA_MODEL_PATH), 'w') as file:
        json.dump(o, file, default=custom_serializer, indent=4)

def dump_c(columns_structure):
    with open(safe_access_path(AppConstants.COLUMN_STRUCTURE_PATH), 'w') as file:
        json.dump(columns_structure, file,default=custom_serializer, indent=4)




