from setuptools import find_packages, setup

package_name = 'prism_topomap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='docker_prism',
    maintainer_email='docker_prism@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'prism_topomap = prism_topomap.prism_topomap_node:main',
            'tf_to_odom = prism_topomap.tf_to_odom:main'
        ],
    },
)
