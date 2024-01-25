import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setuptools.setup(
    name="birdhub",
    version="0.0.1",
    packages=setuptools.find_packages(),
    install_requires=required,
    entry_points={
        "console_scripts": [
            "pgn = birdhub.cli:cli",
        ],
    },
)
