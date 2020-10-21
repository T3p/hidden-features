from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='lrcb',
      packages=[package for package in find_packages()
                if package.startswith('lrcb')],
      install_requires=[
              'numpy',
              'scipy',
              'matplotlib',
              'jupyter',
              'pandas',
              'scikit-learn',
              'tensorboardx'],
      description="Learning representations for contextual bandits",
      author="Matteo Papini",
      url='https://github.com/T3p/lrcb',
      author_email="matteo.papini@polimi.it",
      version="0.1.1")
