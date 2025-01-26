from setuptools import setup, find_packages

setup(
    name="license-plate-case",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fire==0.7.0",
        "ultralytics==8.3.68",
        "tqdm==4.67.1",
        "easyocr==1.7.2"
    ],
    entry_points={
        'console_scripts': [
            'license-plate-case=license_plate_case.cli:main',
        ],
    },
    author="Gabriel Cunha",
    author_email="gsalomaoc@gmail.com",
    description="A tool for license plate detection and tracking",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Gcunhaa/case-computer-vision",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
) 