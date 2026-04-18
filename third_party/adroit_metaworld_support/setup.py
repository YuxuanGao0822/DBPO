from setuptools import find_packages, setup


setup(
    name="adroit_metaworld_support",
    version="0.1.0",
    description="Support package for DBPO Adroit / MetaWorld data generation.",
    packages=find_packages(),
    py_modules=[
        "adroit",
        "stage1_models",
        "transfer_util",
        "utils",
        "vrl3_agent",
    ],
)
