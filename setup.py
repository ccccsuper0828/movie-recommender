"""
CineMatch - Movie Recommendation System
Setup configuration for package installation
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#") and not line.startswith("-")
    ]
else:
    requirements = []

setup(
    name="cinematch",
    version="1.0.0",
    author="CineMatch Team",
    author_email="team@cinematch.example.com",
    description="Intelligent movie recommendation system with multiple ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/cinematch",
    project_urls={
        "Bug Tracker": "https://github.com/your-username/cinematch/issues",
        "Documentation": "https://github.com/your-username/cinematch/docs",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Internet :: WWW/HTTP :: HTTP Servers",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "docs"]),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.11.0",
            "isort>=5.12.0",
            "flake8>=6.1.0",
            "mypy>=1.7.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "cinematch-api=api.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "frontend": ["styles/*.css"],
    },
    zip_safe=False,
)
