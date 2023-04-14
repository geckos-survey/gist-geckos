from setuptools import setup
from setuptools import find_packages
import os

def readme():
    with open('README.md') as file:
        return(file.read())

def versionNumber():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gistPipeline/_version.py')) as versionFile:
        return(versionFile.readlines()[-1].split()[-1].strip("\"'"))

setup(name='gistPipeline',
      version=versionNumber(),
      description='The GIST Framework: A multi-purpose tool for the analysis and visualisation of (integral-field) spectroscopic data',
      long_description_content_type="text/markdown",
      long_description=readme(),
      classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Programming Language :: Python :: 3',
        'Development Status :: 5 - Production/Stable',
        'License :: Other/Proprietary License',
      ],
      url='https://abittner.gitlab.io/thegistpipeline/',
      author='Adrian Bittner',
      author_email='adrian.bittner@eso.org',
      license='Other/Proprietary License',
      packages=find_packages(),
      install_requires=[
        'astropy>=3.1',
        'emcee>=2.2',
        'matplotlib>=3.1',
        'numpy>=1.17',
#        'PyQt6>=5.10',
        'scipy>=1.3',
        'pyyaml>=5.3',
        'vorbin>=3.1',
        'ppxf>=6.7',
        'plotbin>=3.1',
        'printStatus>=1.0',
        'multiprocess>=0.5'
      ],
      python_requires='>=3.11',
      entry_points={
        'console_scripts': [
            'gistPipeline        = gistPipeline.MainPipeline:main'
            ],
        'gui_scripts': [
            'Mapviewer           = gistPipeline.Mapviewer:main',
            'gistPlot_kin        = gistPipeline.plotting.gistPlot_kin:main',
            'gistPlot_lambdar    = gistPipeline.plotting.gistPlot_lambdar:main',
            'gistPlot_gas        = gistPipeline.plotting.gistPlot_gas:main',
            'gistPlot_sfh        = gistPipeline.plotting.gistPlot_sfh:main',
            'gistPlot_ls         = gistPipeline.plotting.gistPlot_ls:main'
         ],
      },
      include_package_data=True,
      zip_safe=False)
