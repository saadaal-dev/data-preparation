from setuptools import setup, find_packages

# Read direct (non-transitive) dependencies from requirements.in.
# requirements.txt is the pinned lockfile for reproducible installs.
def read_requirements():
    with open('requirements.in') as req_file:
        lines = req_file.read().splitlines()
    return [l for l in lines if l and not l.startswith('#')]

setup(
    name='shaqodoon-forecasting',
    version='0.1.0',
    packages=find_packages(include=['shaqodoon_forecasting', 'shaqodoon_forecasting.*']),
    install_requires=read_requirements(),
    python_requires='>=3.8',
)
