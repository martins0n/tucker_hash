from setuptools import setup, find_packages


setup(
    name='tuckerhash',
    version='0.0.1',
    description='Tucker decomposition for image hashing',
    url='https://github.com/martins0n/nla_project_image_hash',
    packages=find_packages(exclude=('tests'))
)