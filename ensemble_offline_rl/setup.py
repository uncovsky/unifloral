from setuptools import setup, find_packages

setup(
    name="ensemble_offline_rl",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium",
    ],
    include_package_data=True,
    python_requires=">=3.8",
)
