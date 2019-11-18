import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="veritorch", # Replace with your own username
    version="0.0.3",
    author="Yixing Guan, Yinan Shen, Shuying Ni, Banruo(Rock) Zhou",
    author_email="yixingguan@fas.harvard.edu, yinanshen@g.harvard.edu, shuying_ni@g.harvard.edu, bzhou@g.harvard.edu",
    description="autodifferentiation package, supporting forward mode only right now",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/veritorch/cs207-FinalProject",
    install_requires = [
        'numpy==1.17.3',
        'pytest==5.2.2'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

