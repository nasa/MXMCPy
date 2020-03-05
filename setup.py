import setuptools

setuptools.setup(
    name="mxmc", 
    version="0.0.1",
    author="Geoffrey F. Bomarito, James E. Warner, Patrick E. Leser, William P. Leser, Luke Morrill",
    description="Multi Model Monte Carlo with Python",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/nasa/mxmc",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
