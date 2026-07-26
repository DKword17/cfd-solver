from setuptools import setup, find_packages

setup(
    name="cfd_solver",
    version="0.2.0",
    description="2D incompressible Navier-Stokes solver with structured mesh, RANS/LES turbulence models, and CUDA GPU acceleration.",
    author="cfd-solver contributors",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.24"],
    extras_require={"dev": ["pytest>=7.0"]},
)
