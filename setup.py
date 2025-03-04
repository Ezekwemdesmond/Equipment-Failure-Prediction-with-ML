from setuptools import find_packages, setup
from typing import List

# Define a function to read requirements from requirements.txt
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements.txt file and returns a list of requirements.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        if "-e ." in requirements:
            requirements.remove("-e .")
    return requirements

# Setup configuration
setup(
    name="EquipmentFailurePrediction",  
    version="0.0.1",  
    author="Ezekwem Desmond",  
    author_email="engrstephdz@gmail.com",  
    description="A machine learning project for predicting equipment failures.",  
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",  
    url="https://github.com/Ezekwemdesmond/Equipment-Failure-Prediction-with-ML",  
    packages=find_packages(where="src"),  # Look for packages in the `src` directory
    package_dir={"": "src"},  # Map the root of the package to the `src` directory
    install_requires=get_requirements("requirements.txt"),  
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",  
    ],
    python_requires=">=3.6",  # Minimum Python version required
)