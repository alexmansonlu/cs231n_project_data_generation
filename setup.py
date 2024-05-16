from setuptools import find_packages, setup
import os

package_name = 'data_generation'

# Function to recursively include all files in a directory
def package_files(directory):
    paths = []
    # Walking through the directory, including all files
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            # Append the file path to paths, making sure to specify the relative path in the package
            full_path = os.path.join(path, filename)
            install_path = os.path.join('share', package_name, 'Worlds', os.path.relpath(path, directory))
            paths.append((install_path, [full_path]))

    return paths

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/metrics.launch.py']),
        # ('share/' + package_name + '/Worlds', ['Worlds/metrics.world']),
    ] + package_files('Worlds'),
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alexmanson',
    maintainer_email='alexmanson_lu@outlook.com',
    description='package for data generation in Gazebo',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gz_render = data_generation.gz_render:main'
        ],
    },
)
