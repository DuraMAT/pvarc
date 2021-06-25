import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pvarc",
    version="0.0.7",
    author="toddkarin",
    author_email="pvtools.lbl@gmail.com",
    description="Analyze anti-reflection coatings on the air-glass interface of a photovoltaic module.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DuraMAT/pvarc",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy','pandas','scipy','matplotlib','tmm','colour-science','tqdm'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)