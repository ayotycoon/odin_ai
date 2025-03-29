import os
import pickle

from future.backports.datetime import datetime
from pandas import Timedelta

from main.config.app_constants import AppConstants
from main.config.logger import global_logger
from main.utils.path_utils import safe_access_path




_loaded_cache = {}
def load_pickle(key:str, expiry:Timedelta = None):
    dir = safe_access_path(f"{AppConstants.APP_PATH}/.temp/cache/{key}.pkl")
    try:

        def is_expired(_date:float):
            if expiry is None:
                return False
            if isinstance(_date, (int, float)):  # If _date is a timestamp
                _date = datetime.fromtimestamp(_date)
            file_date = (_date + expiry)
            return datetime.now() > file_date
        file_modified = None
        file = None
        if key in _loaded_cache:
            file,file_modified = _loaded_cache[key]
        if file_modified is None or is_expired(file_modified) is True:
            with open(dir, "rb") as f:
                file,file_modified = pickle.load(f), datetime.fromtimestamp(os.path.getmtime(dir))
                if is_expired(os.path.getmtime(dir)) is True:
                    return None
                _loaded_cache[key] = file,file_modified
        return file
    except Exception as e:
        global_logger.debug(e, exc_info=True)
        return None

def to_pickle(key, obj):
    # Serialize the object to a file
    with open(safe_access_path(f"{AppConstants.APP_PATH}/.temp/cache/{key}.pkl"), "wb") as f:
        pickle.dump(obj, f)
    global_logger.debug(f"finished converting to pickle {key}")

