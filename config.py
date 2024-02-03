# config.py
import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        # Load variables from .env file
        load_dotenv()

        # Set default values or fallbacks if needed
        self.server_ip = os.getenv("SERVER_IP", "0.0.0.0")
        self.server_port = os.getenv("SERVER_PORT", "5000")

# Instantiate the Config class
config = Config()
