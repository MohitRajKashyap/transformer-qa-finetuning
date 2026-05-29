"""
Setup configuration for Transformer QA Fine-Tuning project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="transformer-qa-finetuning",
    version="1.0.0",
    author="Research Engineering Team",
    author_email="research@example.com",
    description=(
        "Production-grade DistilBERT fine-tuning pipeline "
        "for Extractive Question Answering on SQuAD v1.1"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/transformer-qa-finetuning",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qa-train=src.training.train:main",
            "qa-evaluate=src.training.evaluate:main",
            "qa-predict=src.inference.predict:main",
            "qa-ablation=src.experiments.ablation:main",
        ],
    },
    extras_require={
        "dev": [
            "black",
            "isort",
            "flake8",
            "mypy",
            "pytest",
            "pytest-cov",
        ],
        "wandb": ["wandb>=0.16.0"],
    },
)
