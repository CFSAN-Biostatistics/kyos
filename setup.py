#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    "tensorflow",
    "Biopython",
    "pandas",
    "pysam",
]

test_requirements = [
    "pytest",
]

setup(
    name='kyos',
    version='0.2.0',
    description="Tools for haploid variant calling with Deep Neural Networks.",
    long_description=readme + '\n\n' + history,
    author="Nathan Xue",
    author_email='xue.nathanv1.0@gmail.com',
    url='https://github.com/CFSAN-Biostatistics/kyos',
    packages=[
        'kyos',
    ],
    package_dir={'kyos':
                 'kyos'},
    include_package_data=True,
    install_requires=requirements,
    license="BSD",
    zip_safe=False,
    keywords=['bioinformatics', 'NGS', 'kyos'],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    entry_points={'console_scripts': ['kyos = kyos.cli:main']},
    setup_requires=["pytest-runner"],
    tests_require=test_requirements
)
