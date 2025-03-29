import os
import platform
import pyautogui
import time

from main.config.app_constants import AppConstants
from main.config.logger import global_logger
from main.config.env import Env


class DevUtils:
    @staticmethod
    def is_mac_dev():
        if Env.is_dev and platform.system() == "Darwin":
            return True
        global_logger.error("System must be mac and env must be DEV to run dev utils")
        return False


    @staticmethod
    def run_caffinate():
        if DevUtils.is_mac_dev() is False:
            return
        bash_script_path = f"{AppConstants.APP_PATH}/prevent_mac_sleep.bash.sh"
        global_logger.debug(bash_script_path)
        os.system(f'osascript -e \'tell application "Terminal" to do script "bash {bash_script_path}"\'')
    @staticmethod
    def click_screen():
        t = 7200
        sleep_time = 60
        # Coordinates of the point to click (adjust as needed)
        x, y = 500, 400  # Change these values as per your requirement

        for x in range(0, int(t/sleep_time)):
            # Delay to switch to the target window
            time.sleep(sleep_time)
            # Move the mouse and click
            pyautogui.moveTo(x, y, duration=0.5)
            pyautogui.click()
            global_logger.debug('clicked')




if __name__ == '__main__':
    DevUtils.click_screen()



