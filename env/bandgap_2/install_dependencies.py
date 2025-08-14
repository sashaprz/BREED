#!/usr/bin/env python3
"""
Installation script for Materials Project bandgap extraction dependencies
"""

import subprocess
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip"""
    try:
        logger.info(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        return False

def main():
    """Install all required packages"""
    packages = [
        "mp-api>=0.41.0",
        "pymatgen>=2023.1.1", 
        "pandas>=1.5.0",
        "numpy>=1.21.0"
    ]
    
    logger.info("Installing Materials Project API dependencies...")
    
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        logger.error(f"Failed to install: {', '.join(failed_packages)}")
        logger.info("Try installing manually with:")
        for package in failed_packages:
            logger.info(f"  pip install {package}")
        return False
    else:
        logger.info("All packages installed successfully!")
        logger.info("You can now run: python extract_mp_data.py")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)