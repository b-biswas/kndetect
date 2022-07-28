from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "matplotlib",
    "numpy",
    "pandas",
    "scikit-learn",
    "astropy",
    "tqdm",
    "actsnclass==1.2",
]

setup(
    name="kndetect",
    version="0.0.1",
    author="Biswajit Biswas",
    author_email="biswas@apc.in2p3.fr",
    description="Kilonova detection module for Fink broker",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/b-biswas/kndetect",
    include_package_data=True,
    packages=["kndetect"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    package_data={"kndetect": ["data/*"]},
)
