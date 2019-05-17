from distutils.core import setup

setup(
    name='ml_utils',
    version='0.1.0',
    description='Utilities for machine learning with scikit-learn.',
    author='Stefanie Molin',
    author_email='24376333+stefmolin@users.noreply.github.com',
    license='MIT',
    url='https://github.com/stefmolin/ml-utils',
    packages=['ml_utils'],
    install_requires=[
        'matplotlib>=3.0.3',
        'numpy>=1.16.3',
        'pandas>=0.23.4',
        'scikit-learn>=0.20.3',
        'seaborn>=0.9.0'
    ],
)
