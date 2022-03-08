from setuptools import setup, find_packages


# Use
#   pip install -e .
# to install the project as an editable installation

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='genre_muse',
    version='1.0.0',
    description='A machine learning package for genre prediction.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jaskee1/top-n-genre-classification',
    packages=find_packages(
        exclude=[
            "*.tests",
            "*.tests.*",
            "tests.*",
            "tests"
        ]
    ),
    package_data={
        # If any package contains *.h5 files, include them.
        # This is used to get the trained models that would otherwise be
        # excluded.
        "": ["*.h5"],
    },
    install_requires=[
        'librosa',
        'matplotlib',
        'numpy',
        'pandas',
        # Using cpu version for smaller install size
        'tensorflow-cpu',
    ],
)
