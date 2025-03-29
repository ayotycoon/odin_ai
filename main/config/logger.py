import warnings
warnings.filterwarnings('ignore')
import logging
import os


from colorama import Fore, Style, init

from main.config.app_constants import AppConstants
from main.config.env import Env

# Initialize colorama for cross-platform support
init(autoreset=True)


switch_dict = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,

}

LOG_COLORS = {
    'DEBUG': Fore.BLUE,
    'INFO': Fore.GREEN,
    'WARNING': Fore.YELLOW,
    'ERROR': Fore.RED,
    'CRITICAL': Fore.MAGENTA + Style.BRIGHT
}

# Create a logger
def get_logger(name:str):
    log_file_path = f"{AppConstants.LOG_FOLDER_PATH}/my_log.log"
    _logger = logging.getLogger(name)
    _logger.setLevel(logging.INFO if not switch_dict[Env.LOG_LEVEL] else switch_dict[Env.LOG_LEVEL])
    # Create a file handler

    file_handler = logging.FileHandler(log_file_path)

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)

    def read_last_log_line(module_name=None):
        """Reads the last appended line from the log file."""
        with open(log_file_path, "rb") as file:
            file.seek(0, os.SEEK_END)  # Move to the end of the file
            file_size = file.tell()

            if file_size == 0:
                return "Log file is empty."

            # Read last line efficiently
            file.seek(max(file_size - 1024, 0), os.SEEK_SET)  # Go back max 1024 bytes
            lines = file.readlines()
            if not lines:
                return "No logs found."
            lines = [line.decode("utf-8").strip() for line in lines]
            if module_name:
                filtered_lines = [line for line in lines if module_name in line]
                if filtered_lines:
                    return filtered_lines[-1]  # Return the last matching line
                else:
                    return f"No logs found for module: {module_name}"
            else:
                return lines[-1]
    _logger.read_last_log_line = read_last_log_line

    return _logger

global_logger = get_logger(__name__)
global_logger.info(f"working dir = {os.getcwd()}")
global_logger.info(f"env={Env.ENV};LOG_LEVEL={Env.LOG_LEVEL}")

def clear_console():
    if os.environ.get('TERM'):
        os.system('cls' if os.name == 'nt' else 'clear')


