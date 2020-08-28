import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("requirements.txt") as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="classitransformers",
    packages=['classitransformers','electra'],
    version="0.0.1",
    author="FMR LLC",
    author_email="classitransformers@fmr.com",
    description="An abstract library for implementing text classification tasks based on various transformers based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fidelity/classitransformers",
    install_requires=required,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    keywords='NLP language_models text_classification bert electra roberta distilbert albert',
    
    project_urls={
        "Source": "https://github.com/fidelity/classitransformers"
    }
)

