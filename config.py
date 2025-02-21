import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/FYP_Backend")
    SECRET_KEY = os.getenv("SECRET_KEY", "dev_secret_key")
    SESSION_TYPE = "mongodb"
    SESSION_TYPE = "filesystem"  # Can be 'redis', 'memcached', 'mongodb', etc.
    SESSION_PERMANENT = False
    SESSION_USE_SIGNER = True
    SESSION_FILE_DIR = "./flask_session_data"  # Directory for storing session files
    SECRET_KEY = "your_secret_key"  # Required for session security