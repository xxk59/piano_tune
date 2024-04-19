import setuptools

def readme():
    with open('README.md') as f:
        return f.read()

setuptools.setup(
    name="pypianotune",
    version="0.1.0",
    description="A Python module to convert music notes into piano tune.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    author="Kelvin Xu",
    author_email="xxk59@hotmail.com",
    url="https://github.com/xxk59/piano_tune",
    license="MIT",
    packages=["pypianotune"],
    setup_requires=[
        "numpy"
    ],
    install_requires=[
        "numpy"
    ],
)