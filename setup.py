from setuptools import setup
from nmr.version import __version__

setup(
    name='nmr',
    url='https://github.com/Piers14/numerai',
    author='Piers Hinds',
    author_email='pmxph7@nottingham.ac.uk',
    packages=['nmr'],
    install_requires=['numpy', 'torch', 'pandas', 'numerapi', 'sklearn', 'scipy'],
    tests_require=['pytest'],
    version=__version__,
    license='MIT',
    description='Helpers for numerai'
)