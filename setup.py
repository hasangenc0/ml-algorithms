import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ml",
    version="0.0.1",
    author="Hasan Genc",
    author_email="hasangencx@hotmail.com",
    description="Basic implemtations of ML algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hasangenc0/ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
)