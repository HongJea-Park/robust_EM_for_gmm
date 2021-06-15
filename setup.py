from setuptools import find_packages, setup


install_requires = open("requirements.txt").read().strip().split("\n")
dev_requires = open("dev-requirements.txt").read().strip().split("\n")
test_requires = open("test-requirements.txt").read().strip().split("\n")

extras = {
    "dev": dev_requires + test_requires,
    "test": test_requires
}

extras["all_extras"] = sum(extras.values(), [])

setup(
    name="robustgmm",
    version='1.0.0',
    description="A robust EM clustering algorithm for Gaussian mixture models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="HongJea Park",
    author_email="hongjea.park@gmail.com",
    url="https://github.com/HongJea-Park/robust_EM_for_gmm.git",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6",
    install_requires=install_requires,
    extras_require=extras,
    setup_requires=['pytest-runner']
)
