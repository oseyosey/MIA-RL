from setuptools import setup, find_packages

setup(
    name="adra",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Core dependencies
        "evaluate",
        "transformers",
        "torch",
        
        # Visualization and data analysis
        "seaborn",
        "pandas",
        
        # Data processing
        "pyarrow>=19.0.0",
        
        # Required by utils
        "numpy",
        "scikit-learn",
    ],
    python_requires=">=3.10",
)
