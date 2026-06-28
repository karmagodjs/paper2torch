from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="paper2torch",
    version="0.1.0",
    author="Dhruv Kumar",
    author_email="rafftarsingh7982@gmail.com",
    description="Convert research papers to PyTorch code using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karmagodjs/paper2torch",
    packages=find_packages(),
    install_requires=[
        "groq",
        "pymupdf",
        "click",
        "rich",
        "python-dotenv",
    ],
    entry_points={
        "console_scripts": [
            "paper2torch=paper2torch.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)