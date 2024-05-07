from setuptools import setup, find_packages

setup(
    name='basketball_gym',
    version='1.0',
    description='A custom Gym environment for basketball simulation',
    author='Windsor Nguyen',
    author_email='mn4560@princeton.edu',
    packages=find_packages(),
    install_requires=[
        'gym',
        'mujoco',
        'numpy'
    ],
    entry_points={
        'gym.envs': [
            'BasketballEnv-v0 = basketball_gym.basketball_env:BasketballEnv',
        ],
    },
)
