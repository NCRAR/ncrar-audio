from pathlib import Path
from setuptools import find_packages, setup


requirements = [
    'numpy',
    'scipy',
    'pandas',
    'python-osc',
    'sounddevice',
]


extras_require = {
    'docs': ['sphinx', 'sphinx_rtd_theme', 'pygments-enaml'],
    'test': ['pytest', 'pytest-benchmark'],
}


# Get version number
version_file = Path(__file__).parent / 'ncrar_audio' / '__init__.py'
for line in version_file.open():
    if line.strip().startswith('__version__'):
        version = line.split('=')[1].strip().strip('\'')
        break
else:
    raise RuntimeError('Could not determine version')


setup(
    name='ncrar-audio',
    author='NCRAR audio development team',
    author_email='info@bradburan.com',
    install_requires=requirements,
    extras_require=extras_require,
    packages=find_packages(),
    include_package_data=True,
    description='NCRAR audio tools',
    entry_points={
        'console_scripts': [],
    },
)
