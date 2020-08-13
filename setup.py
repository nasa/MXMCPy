import setuptools

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

authors = [
    "Geoffrey F. Bomarito",
    "James E. Warner",
    "Patrick E. Leser",
    "William P. Leser",
    "Luke Morrill"
]

setuptools.setup(
    name="mxmcpy",
    version="1.0",
    author=", ".join(authors),
    author_email="",
    description="Multi Model Monte Carlo with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/MXMCPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.7',
    install_requires=['numpy',
                      'pandas',
                      'scipy',
                      'torch',
                      'h5py']
)
