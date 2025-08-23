from setuptools import setup, find_packages

setup(
    name="cdvae",
    version="0.1.0",
    description="Crystal Diffusion Variational Autoencoder",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "pytorch-lightning",
        "hydra-core",
        "omegaconf",
        "wandb",
        "numpy",
        "pandas",
        "pymatgen",
        "ase",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "seaborn",
    ],
)