from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="zarr_converter",
    version="0.1.0",
    description="Convert microscopy images (TIFF, CZI, LIF, H5) to pyramidal OME-Zarr with metadata.",
    author="Your Name",
    author_email="you@example.com",
    packages=find_packages(),
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "zarr-convert=zarr_converter.cli:convert"
        ]
    },
    python_requires=">=3.8",
)