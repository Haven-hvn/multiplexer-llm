#!/usr/bin/env python3
"""Setup script for multiplexer-llm package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="multiplexer-llm",
    version="0.4.0",
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
 "Programming Language :: Python ::3.10",
 "Programming Language :: Python ::3.11",
 "Programming Language :: Python ::3.12",
 "Programming Language :: Python ::3.13",
 "Programming Language :: Python ::3.14",
 "Topic :: Software Development :: Libraries :: Python Modules",
 "Topic :: Scientific/Engineering :: Artificial Intelligence",
 ],
 python_requires=">=3.10",
 install_requires=requirements,
    extras_require={
        "dev": [
            "pytest==9.1.1",
            "pytest-asyncio==1.4.0",
            "pytest-mock==3.16.0",
            "black==26.10.1",
            "isort==9.0.2",
            "mypy==2.4.0",
            "flake8==7.4.1",
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
