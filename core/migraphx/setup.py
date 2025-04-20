# Author: Stanley Chisango (Scooter Lacroix)
# Email: scooterlacroix@gmail.com
# GitHub: https://github.com/scooter-lacroix
# X: https://x.com/scooter_lacroix
# Patreon: https://patreon.com/ScooterLacroix
# 
# If this code saved you time, consider buying me a coffee! ☕
# "Code is like humor. When you have to explain it, it's bad!" - Cory House

from setuptools import setup, find_packages

setup(
    name="migraphx",
    version="2.12.0",
    packages=find_packages(),
    package_data={
        'migraphx': ['*.so'],
    },
    description="AMD MIGraphX Python bindings",
    author="AMD",
    author_email="",
    url="https://github.com/ROCm/AMDMIGraphX",
)
