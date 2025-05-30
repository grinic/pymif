from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pymif",
    version="0.1.0",
    description="Convert microscopy images (TIFF, CZI, LIF, H5) to pyramidal OME-Zarr with metadata.",
    author="Nicola Gritti",
    author_email="nicola.gritti@embl.es",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)