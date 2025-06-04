from setuptools import setup, find_packages

setup(
    name="PhDMelodySet",
    version="0.1",
    author="David Whyatt",
    author_email="dmw56@cam.ac.uk",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/davidwhyatt/PhDMelodySet",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'scipy>=1.7.0',
        'tqdm>=4.65.0',
        'pretty_midi>=0.2.10',
    ],
) 