from setuptools import setup, find_packages

setup(
    name='blue_recon',
    version='0.1.0',
    packages=find_packages(),
    install_requires=['setuptools'],
    entry_points={
        'console_scripts': [
            'bridge_node = blue_recon.bridge_node:main',
        ],
    },
)
