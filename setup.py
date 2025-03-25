from setuptools import setup, find_packages

setup(
    name='Attention_is_all_you_need',
    version='0.1.0',
    description='',
    author='Fadi Benzaima',
    packages=find_packages(where='Attention_is_all_you_need'),
    package_dir={'': 'Attention_is_all_you_need'},
    install_requires=[
        'torch',
        'datasets',
        'transformers',
        'tokenizers',
        'einops'
    ],
)