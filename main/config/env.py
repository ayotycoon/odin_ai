from dotenv import load_dotenv
import os

env_str = "DEV" if not os.environ.get("ENV") else os.environ.get("ENV")

load_dotenv()
if env_str:
    load_dotenv(".env."+env_str.lower(), override=True)

class Env:
    APP_NAME: str = os.environ.get("APP_NAME")
    MONGO_DB_URI: str = os.environ.get("MONGO_DB_URI")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # The ID and range of a sample spreadsheet.
    SAMPLE_SPREADSHEET_ID: str = os.environ.get("SAMPLE_SPREADSHEET_ID")
    ODIN_URL: str = os.environ.get("ODIN_URL")
    ENV =  env_str
    DISABLE_CACHE = os.environ.get("DISABLE_CACHE") == 'True'
    is_dev =  env_str == "DEV"
    is_prod =  env_str == "PROD"

