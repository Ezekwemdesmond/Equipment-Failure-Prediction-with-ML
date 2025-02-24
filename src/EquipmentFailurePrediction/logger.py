import os
import sys
import logging

# Define the log format
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Define the log directory and log file path
log_dir = "logs"
log_filepath = os.path.join(log_dir,"running_logs.log")
os.makedirs(log_dir, exist_ok=True)   # Create the log directory if it doesn't exist


# Configure the custom logger
logging.basicConfig(
    level= logging.INFO,
    format= logging_str,

    handlers=[
        logging.FileHandler(log_filepath),   # Log to a file
        logging.StreamHandler(sys.stdout)    # Log to console
    ]
)

# Create a custom logger
logger = logging.getLogger("EquipmentFailurePredictionLogger")