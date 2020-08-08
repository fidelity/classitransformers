import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="classitransformers",
    packages=['classitransformers'],
    version="0.0.1",
    description="An abstract library for implementing text classification tasks based on various transformers based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fidelity/classitransformers",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache License 2.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    keywords='bert electra classification roberta distilbert albert',
    author='FMR LLC',
)

