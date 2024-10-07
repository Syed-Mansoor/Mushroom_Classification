import logging 
import os
from from_root import from_root
from datetime import datetime

# Generate log file name
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_dir = 'logs'
logs_path = os.path.join(from_root(), log_dir, LOG_FILE)  # Ensure from_root() is called

# Create logs directory if it doesn't exist
os.makedirs(os.path.join(from_root(), log_dir), exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(logs_path)  # File output
    ]
)

# Example logging
logging.info("Logging setup complete.")