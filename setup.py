"""
Setup configuration for ML Swell Noise Suppression project.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ml-swell-noise-suppression",
    version="0.1.0",
    author="Anshul Choudhary",
    author_email="24mc0013@iitism.ac.in",
    description="ML-assisted swell noise suppression in towed marine seismic data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AnshulChoudhary-iitism/Machine_learning_noise_supression",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "ml-noise-gui=src.ui.main_window:main",
        ],
    },
)
