from __future__ import print_function
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EQTransformer",
    version="0.1.11",
    author="S. Mostafa Mousavi",
    author_email="smousavi05@gmail.com",
    description="A python package for making and using attentive deep-learning models for earthquake signal detection and phase picking.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smousavi05/EQTransformer",
    packages=setuptools.find_packages(),
    keywords='Seismology, Earthquakes Detection, P&S Picking, Deep Learning, Attention Mechanism',
    install_requires=['pytest', 'tensorflow==2.0.1', 'keras', 'matplotlib', 'scipy', 'pandas', 'tqdm', 'Obspy', 'h5py'], 
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',      
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3',
)


