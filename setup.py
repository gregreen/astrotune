import os
import sys
import shutil
from setuptools import setup, find_packages, find_namespace_packages
from setuptools.command.install import install

setup(name="astrotune",
      version='1.0.0',
      description='Harmonize sets of stellar parameters',
      author='Gregory M. Green',
      author_email='gregorymgreen@gmail.com',
      url='https://github.com/gregreen/astrotune',
      requires=['numpy','astropy(>=4.0)','scipy','tensorflow'],
      zip_safe = False,
      include_package_data=True,
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"},
)
