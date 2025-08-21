from setuptools import setup, find_packages

setup(
    name='neurogenpredict',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'streamlit',
    ],
    author='Dany Mukesha',
    description="A Genetic Risk Assessment Tool for Alzheimer's Disease",
    long_description="This tool combines weighted polygenic risk scores, pathway-based analysis, and ensemble ML for interpretable AD risk prediction.",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
