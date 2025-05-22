import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymif",
    version="0.0.1",
    author="Nicola Gritti",
    author_email="gritti@embl.es",
    description="A set of functions useful for MIF users.",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib==3.8",
        "numpy==1.26",
        "scipy==1.11",
        "pandas==2.1",
        "scikit-image==0.22",
        "seaborn==0.13",
        "imagecodecs==0.0.1",
        # "PyQt5",
        "scikit-learn==1.3",
        "tqdm==4.66",
    ],
    python_requires='>=3.6',
    zip_safe=False 
    )
