from datetime import datetime

from main.config.logger import global_logger

def seconds_formatter(total_seconds:float):
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    str_res = f"{seconds}s"
    if minutes > 0:
        str_res=f"{minutes}m {str_res}"
    if hours > 0:
        str_res=f"{hours}h {str_res}"
    return str_res



def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()  # Record the start time
        global_logger.debug(f"{func.__name__} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)  # Call the actual function
        end_time = datetime.now()  # Record the end time
        # logger.debug(f"{func.__name__} ended at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        diff = (end_time - start_time)

        global_logger.info(f"{func.__name__} executed in {seconds_formatter(diff.total_seconds())}")
        return result
    return wrapper

class ElapsedTime:
    def __init__(self, replace = False):
        # self.start_time = None
        # self.end_time = None
        self.replace = replace
        self.reset()
    def reset(self):
        self.end_time = None
        self.start_time = datetime.now()

    def log(self):
        self.end_time = datetime.now()
        diff = (self.end_time - self.start_time)
        if self.replace:
            self.start_time = self.end_time
        return seconds_formatter(diff.total_seconds())

    def log_if_true(self, str_t, condition):
        if condition:
            return f"{str_t} {self.log()}"
        return ""
