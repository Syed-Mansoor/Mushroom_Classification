import setuptools

with open('README.md','r',encoding = 'utf-8') as f:
    long_description = f.read()


__version__ = '0.0.0'

REPO_NAME = 'mushroom_classification'
AUTHOR_USER_NAME = 'syed-mansoor'
SRC_REPO = 'mushroom'
AUTHOR_EMAIL = 'saeedmansoor56@gmail.com'

setuptools.setup(
    name = SRC_REPO,
    version=__version__,
    author= AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description='A small python package for mushroom classification',
    long_description=long_description,
    long_description_content = 'test/markdown',
    url = f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues',
    project_urls={
        'Bug Tracker': f'https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues'
    },
    package_dir={'': 'mushroom'},
    packages=setuptools.find_packages(where='mushroom')
)