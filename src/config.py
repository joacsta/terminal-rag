from os import getenv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


PATH_BASE = str(Path(__file__).parent.parent / "resources")
PATH_DB = str(Path(__file__).parent.parent / "database")
API_KEY = getenv("GOOGLE_API_KEY", None)
MODEL_EMBEDDING = getenv("MODEL_EMBEDDING", None)
