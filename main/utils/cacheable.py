from main.config.logger import global_logger
from main.config.env import Env
from main.utils.datetime_utils import get_time_delta
from main.utils.pickle_utils import to_pickle, load_pickle


def cacheable(expiry: str = None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if Env.DISABLE_CACHE:
                return func(*args, **kwargs)

            key = f"{func.__name__}"
            storage_key = "@cacheable_"+key

            # if len(args) > 0 and "name" in args[0]:
            #     storage_key = storage_key + "#"+args[0].name

            result = load_pickle(storage_key,get_time_delta(expiry))
            cache_found = result is not None
            global_logger.debug(f"@cacheable {func.__name__}#{cache_found}")

            if cache_found is False:
                result = func(*args, **kwargs)
                to_pickle(storage_key,result)
            return result

        return wrapper
    return decorator

def cacheable_callback(full_key, func, *args, **kwargs):
    func_name = func.__name__
    storage_key = "@cacheable_" + full_key
    result = load_pickle(storage_key,get_time_delta(None))
    cache_found = result is not None
    global_logger.debug(f"@cacheable {func_name}#{cache_found}")

    if cache_found is False:
        result = func(*args, **kwargs)
        to_pickle(storage_key,result)
    return result
