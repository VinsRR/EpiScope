
from setuptools import setup, find_packages

setup(
    name="episcope",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "fastapi",
        "pydantic",
        "typer",
        "numpy",
        "scikit-learn",
        "pandas",
        "python-dotenv",
        "torch",
        "transformers",
        "faiss-cpu",
        "unstructured",
        # "grobid-client",
        "ollama",
        # "python-dotenv",
        "pymongo",
    ],
    entry_points={
        "console_scripts": [
            "episcope=episcope.episcope:app",
        ],
    },
)
