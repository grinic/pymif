from setuptools import setup, find_packages

# Read dependencies from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="pymif",
    version="0.1.1",
    description="A collection of functions in Python for MIF users.",
    author="Nicola Gritti",
    author_email="nicola.gritti@embl.es",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
)