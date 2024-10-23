from setuptools import setup, find_packages

setup(
    name="binarysed",
    version="0.1.0",
    author="Jackie Blaum",
    author_email="jackie.blaum@gmail.com",
    description="A package for generating binary SEDs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jackieblaum/binarysed",
    packages=find_packages(),  # Automatically find and include all packages
    install_requires=[
        "numpy",
        "pandas",
        "astropy",
        "matplotlib",
        "scipy",
        "extinction",
        "pyphot",
        "dustmaps",
        "pystellibs",
    ],  # Dependencies required for your package
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify Python version compatibility
)
