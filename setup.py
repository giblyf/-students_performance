from setuptools import find_packages, setup


setup(  
    name='student performance',
    version='0.0.1',
    author='Bulat Dzhamaliddinov',
    author_email='dzhamaliddinovbualt@gmail.com',
    packages=find_packages(),
    install_requires=['pandas', 'numpy', 'seaborn']
)