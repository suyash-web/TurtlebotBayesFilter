from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ras598_assignment_3'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        # Config / RViz / map
        (os.path.join('share', package_name, 'config'),
            glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='RAS598 Student',
    maintainer_email='student@example.com',
    description='Assignment 3 – Discrete Bayes Filter',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bayes_filter = ras598_assignment_3.bayes_filter:main',
        ],
    },
)
