from setuptools import setup, find_packages

setup(
    name="tableref",
    version="0.1.0",
    packages=find_packages(),
    description="A package for extracting tables from scientific papers and matching references within them.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="EpiScope Project",
    url="https://github.com/user/episcope",  # Replace with actual URL
    install_requires=[
        "numpy",
        "pandas",
        "requests",
        "tqdm",
        "rapidfuzz",
        "ollama",
        "nltk",
        "PyMuPDF",
        "layoutparser",
        "opencv-python",
        "img2table",
    ],
    extras_require={
        "torch": [
            "torch",
            "transformers",
            "scikit-learn",
        ],
        "gmft": [
            "gmft",
            "gmft-pymupdf",
        ],
        "dev": [
            "pytest",
            "black",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
