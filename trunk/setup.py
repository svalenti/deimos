from distutils.core import setup
from os import path
from glob import glob

setup(
    name='deimos',
    version='1.0.0',
    author='S. Valenti',
    author_email='stfn.valenti@gmail.com',
    scripts=glob(path.join('bin', '*.py')),
    url='',
    license='LICENSE.txt', 
    description='deimos is a package for spectra reduction',
    long_description=open('README.txt').read(),
    requires=['numpy','astropy','matplotlib'],
    packages=['deimos'],
    package_dir={'': 'src/'},
    package_data={'deimos': ["standard/*txt","standard/*dat","resources/*/*/*","resources/*/*"]}
)
