import os
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,   # Set the logging level to INFO
    format='[%(asctime)s]: %(message)s:')  # Define the log format

# Define the project name and list of files and folders to create
project_name = 'EquipmentFailurePrediction'

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "configs/config.yaml",
    "configs/params.yaml",
    "requirements.txt",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "templates/index.html",
    "templates/result.html"
]

def create_project_structure():
    """
    Creates the directory structure and files for the ML project
    """
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, filename = os.path.split(filepath)
        
        # Create directory if it doesn't exist
        if filedir:
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"Created directory: {filedir}")
        
        # Create empty file if it doesn't exist or has zero size
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            with open(filepath, "w") as f:
                pass
            logging.info(f"Created empty file: {filepath}")
        else:
            logging.info(f"File already exists: {filepath}")


if __name__ == "__main__":
    logging.info(f"Creating project structure for {project_name}")
    create_project_structure()
    logging.info(f"Project structure created successfully")