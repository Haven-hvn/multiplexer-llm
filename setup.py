#!/usr/bin/env python3
"""Setup script for multiplexer-llm package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multiplexer-llm",
    version="0.1.3",
    author="Havencto",
    author_email="officialhavennetwork@gmail.com",
    description="A multiplexer for Large Language Model APIs built on the OpenAI SDK. It combines quotas from multiple models and automatically uses fallback models when the primary models are rate limited.",
    long_description=long_description,
    long_description_content_type="text/markdown",
 url="https://github.com/Haven-hvn/multiplexer-llm",
 packages=find_packages(),
 include_package_data=True,
 data_files=[("", ["requirements.txt"])],
 classifiers=[
 "Development Status ::4 - Beta",
 "Intended Audience :: Developers",
 "License :: OSI Approved :: MIT License",
 "Operating System :: OS Independent",
 "Programming Language :: Python ::3",
 "Programming Language :: Python ::3.8",
 "Programming Language :: Python ::3.9",
 "Programming Language :: Python ::3.10",
 "Programming Language :: Python ::3.11",
 "Programming Language :: Python ::3.12",
 "Topic :: Software Development :: Libraries :: Python Modules",
 "Topic :: Scientific/Engineering :: Artificial Intelligence",
 ],
 python_requires=">=3.8",
 install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
    },
    keywords=[
        "openai",
        "multiplexer",
        "llm",
        "rate-limit",
        "api",
        "ai",
        "machine-learning",
        "chatgpt",
        "claude",
        "gemini",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Haven-hvn/multiplexer-llm/issues",
        "Source": "https://github.com/Haven-hvn/multiplexer-llm",
        "Documentation": "https://github.com/Haven-hvn/multiplexer-llm#readme",
    },
)
